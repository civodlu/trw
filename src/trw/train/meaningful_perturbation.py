import logging
import collections
import functools
from trw.train import guided_back_propagation
from trw.train import outputs as outputs_trw
from trw.train import utilities
from torch.nn import functional as F
from trw.train.filter_gaussian import FilterGaussian
from trw.train.upsample import upsample as upsample_fn
import torch
logger = logging.getLogger(__name__)


def total_variation_norm_2d(x, beta):
    assert len(x.shape) == 4, 'expeted N * C * H * W format!'
    assert x.shape[1] == 1, 'single channel only tested'
    row_grad = torch.mean(torch.abs((x[:, :, :-1, :] - x[:, :, 1:, :])).pow(beta))
    col_grad = torch.mean(torch.abs((x[:, :, :, :-1] - x[:, :, :, 1:])).pow(beta))
    return row_grad + col_grad


def total_variation_norm_3d(x, beta):
    assert len(x.shape) == 5, 'expeted N * C * D * H * W format!'
    assert x.shape[1] == 1, 'single channel only tested'
    depth_grad = torch.mean(torch.abs((x[:, :, :-1, :, :] - x[:, :, 1:, :, :])).pow(beta))
    row_grad = torch.mean(torch.abs((x[:, :, :, :-1, :] - x[:, :, :, 1:, :])).pow(beta))
    col_grad = torch.mean(torch.abs((x[:, :, :, :, :-1] - x[:, :, :, :, 1:])).pow(beta))
    return row_grad + col_grad + depth_grad


def total_variation_norm(x, beta):
    """
    Calculate the total variation norm

    Args:
        x: a tensor with format (samples, components, dn, ..., d0)
        beta:

    Returns:
        a scalar
    """
    if len(x.shape) == 4:
        return total_variation_norm_2d(x, beta)
    elif len(x.shape) == 5:
        return total_variation_norm_3d(x, beta)
    else:
        raise NotImplemented()


def default_optimizer(params, nb_iters, learning_rate=0.1):
    """
    Create a default optimizer for :class:`trw.train.MeaningfulPerturbation`

    Args:
        params: the parameters to optimize
        nb_iters: the number of iterations
        learning_rate: the default learning rate

    Returns:
        a tuple (:class:`torch.optim.Optimizer`, :class:`torch.optim.lr_scheduler._LRScheduler`)
    """
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=nb_iters // 3, gamma=0.1)
    return optimizer, scheduler


def create_inputs(batch, modified_input_name, modified_input):
    """
    Create the model inputs depending on whether the input is a dictionary or tensor
    """
    if isinstance(batch, torch.Tensor):
        return modified_input
    elif isinstance(batch, collections.Mapping):
        new_batch = {}
        for feature_name, feature_value in batch.items():
            if feature_name == modified_input_name:
                new_batch[feature_name] = modified_input
            else:
                new_batch[feature_name] = feature_value
        return new_batch
    else:
        raise NotImplemented()


def default_information_removal_smoothing(image, blurring_sigma=5, blurring_kernel_size=23):
    """
    Default information removal (smoothing).

    Args:
        image: an image
        blurring_sigma: the sigma of the blurring kernel used to "remove" information from the image
        blurring_kernel_size: the size of the kernel to be used. This is an internal parameter to approximate the gaussian kernel. This is exposed since
            in 3D case, the memory consumption may be high and having a truthful gaussian blurring is not crucial.

    Returns:
        a smoothed image
    """
    logger.info('default_perturbate_smoothed: default_perturbate_smoothed, blurring_sigma={}, blurring_kernel_size={}'.format(
        blurring_sigma, blurring_kernel_size
    ))
    gaussian_filter = FilterGaussian(input_channels=image.shape[1], sigma=blurring_sigma, nb_dims=len(image.shape) - 2, kernel_sizes=blurring_kernel_size, device=image.device)
    blurred_img = gaussian_filter(image)
    return blurred_img


class MeaningfulPerturbation:
    """
    Implementation of "Interpretable Explanations of Black Boxes by Meaningful Perturbation", arXiv:1704.03296

    Handle only 2D and 3D inputs. Other inputs will be discarded.

    Deviations:
    - use a global smoothed image to speed up the processing
    """
    def __init__(
            self,
            model,
            iterations=150,
            l1_coeff=0.01,
            tv_coeff=0.2,
            tv_beta=3,
            noise=0.2,
            model_output_postprocessing=functools.partial(F.softmax, dim=1),
            mask_reduction_factor=8,
            optimizer_fn=default_optimizer,
            information_removal_fn=default_information_removal_smoothing,
            export_fn=None,
    ):
        """

        Args:
            model: the model
            iterations: the number of iterations optimization
            l1_coeff: the strength of the penalization for the size of the mask
            tv_coeff: the strength of the penalization for "unnatural" artefacts
            tv_beta: exponentiation of the total variation
            noise: the amount of noise injected at each iteration.
            model_output_postprocessing: function to post-process the model output
            mask_reduction_factor: the size of the mask will be `mask_reduction_factor` times smaller than the input. This is used as regularization to
                remove "unnatural" artifacts
            optimizer_fn: how to create the optimizer
            export_fn: if not None, a function taking (iter, perturbated_input_name, perturbated_input, mask) will be called each iteration (e.g., for debug purposes)
            information_removal_fn: information removal function (e.g., the paper originally used a smoothed image to remove objects)
        """
        self.model = model
        self.iterations = iterations
        self.l1_coeff = l1_coeff
        self.tv_coeff = tv_coeff
        self.tv_beta = tv_beta
        self.noise = noise
        self.model_output_postprocessing = model_output_postprocessing
        self.mask_reduction_factor = mask_reduction_factor
        self.optimizer_fn = optimizer_fn
        self.information_removal_fn = information_removal_fn
        self.export_fn = export_fn

    def __call__(self, inputs, target_class_name, target_class=None):
        """

        Args:
            inputs: a tensor or dictionary of tensors. Must have `require_grads` for the inputs to be explained
            target_class: the index of the class to explain the decision. If `None`, the class output will be used
            target_class_name: the output node to be used. If `None`:
                * if model output is a single tensor then use this as target output

                * else it will use the first `OutputClassification` output

        Returns:
            a tuple (output_name, dictionary (input, explanation mask))
        """
        logger.info('started MeaningfulPerturbation ...')
        logger.info('parameters: iterations={}, l1_coeff={}, tv_coeff={}, tv_beta={}, noise={}, mask_reduction_factor={}'.format(
            self.iterations, self.l1_coeff, self.tv_beta, self.tv_beta, self.noise, self.mask_reduction_factor
        ))
        self.model.eval()  # make sure we are in eval mode

        inputs_with_gradient = dict(guided_back_propagation.GuidedBackprop.get_floating_inputs_with_gradients(inputs))
        if len(inputs_with_gradient) == 0:
            logger.error('MeaningfulPerturbation.__call__: failed. No inputs will collect gradient!')
            return None
        else:
            logger.info('MeaningfulPerturbation={}'.format(inputs_with_gradient.keys()))

        outputs = self.model(inputs)
        if target_class_name is None and isinstance(outputs, collections.Mapping):
            for output_name, output in outputs.items():
                if isinstance(output, outputs_trw.OutputClassification):
                    logger.info('output found={}'.format(output_name))
                    target_class_name = output_name
                    break
        output = MeaningfulPerturbation._get_output(target_class_name, outputs, self.model_output_postprocessing)
        logger.info('original model output={}'.format(utilities.to_value(output)))

        if target_class is None:
            target_class = torch.argmax(output, dim=1)
            logger.info('target_class by sample={}, value={}'.format(target_class, output[:, target_class]))
        else:
            logger.info('target class='.format(target_class))

        # construct our gradient target
        model_device = utilities.get_device(self.model, batch=inputs)
        nb_samples = utilities.len_batch(inputs)

        masks_by_feature = {}
        for input_name, input_value in inputs_with_gradient.items():
            img = input_value.detach()  # do not keep the gradient! This will be recorded by Callback_explain_decision
            if len(img.shape) != 4 and len(img.shape) != 5:
                # must be (Sample, channel, Y, X) or (Sample, channel, Z, Y, X) input
                logging.info('input={} was discarded as the shape={} do not match (Sample, channel, Y, X) or (Sample, channel, Z, Y, X)'.format(input_name, img.shape))
                continue

            logger.info('processing feature_name={}'.format(input_name))

            # must have a SINGLE channel and decreased dim for all others
            mask_shape = [nb_samples, 1] + [d // self.mask_reduction_factor for d in img.shape[2:]]
            logging.info('mask size={}'.format(mask_shape))
            mask = torch.ones(mask_shape, requires_grad=True, dtype=torch.float32, device=model_device)

            # https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351
            blurred_img = self.information_removal_fn(img)
            optimizer, scheduler = self.optimizer_fn([mask], self.iterations)

            assert self.iterations > 0
            c_start = 0.0
            for i in range(self.iterations):
                optimizer.zero_grad()

                upsampled_mask = upsample_fn(mask, img.shape[2:])
                # make the same number of channels for the mask as there is in the image
                upsampled_mask = upsampled_mask.expand([upsampled_mask.shape[0], img.shape[1]] + list(img.shape[2:]))

                # Use the mask to perturbated the input image.
                perturbated_input = img.mul(upsampled_mask) + blurred_img.mul(1 - upsampled_mask)

                noise = torch.zeros(img.shape, device=model_device)
                noise.normal_(mean=0, std=self.noise)

                perturbated_input = perturbated_input + noise
                batch = create_inputs(inputs, input_name, perturbated_input)
                outputs = self.model(batch)

                output = MeaningfulPerturbation._get_output(target_class_name, outputs, self.model_output_postprocessing)
                assert len(output.shape) == 2, 'expected a `N * nb_classes` output'

                # mask = 1, we keep original image. Mask = 0, replace with blurred image
                # we want to minimize the number of mak voxels with 0 (l1 loss), keep the
                # mask smooth (tv + mask upsampling), finally, we want to decrease the
                # probability of `target_class`
                l1 = self.l1_coeff * torch.mean(torch.abs(1 - mask))
                tv = self.tv_coeff * total_variation_norm(mask, self.tv_beta)
                c = output[:, target_class]
                loss = l1 + tv + c

                loss.backward()
                optimizer.step()
                scheduler.step()

                # Optional: clamping seems to give better results
                mask.data.clamp_(0, 1)

                if i == 0:
                    c_start = utilities.to_value(c)

                if i % 20 == 0:
                    logger.info('iter={}, total_loss={}, l1_loss={}, tv_loss={}, c_loss={}'.format(
                        i,
                        utilities.to_value(loss),
                        utilities.to_value(l1),
                        utilities.to_value(tv),
                        utilities.to_value(c),
                    ))

                    if self.export_fn is not None:
                        self.export_fn(i, input_name, perturbated_input, upsampled_mask)

            logger.info('class loss start={}, end={}'.format(c_start, utilities.to_value(c)))
            logger.info('final output={}'.format(utilities.to_value(output)))

            masks_by_feature[input_name] = {
                'mask': 1.0 - utilities.to_value(upsampled_mask),
                'perturbated_input': utilities.to_value(perturbated_input),
                'smoothed_input': utilities.to_value(blurred_img),
                'loss_c_start': c_start,
                'loss_c_end': utilities.to_value(c),
                'output_end': utilities.to_value(output),
            }

        return target_class_name, masks_by_feature

    @staticmethod
    def _get_output(target_class_name, outputs, postprocessing):
        if target_class_name is not None:
            output = outputs[target_class_name]
            if isinstance(output, outputs_trw.Output):
                output = output.output
        else:
            output = outputs

        assert isinstance(output, torch.Tensor)
        output = postprocessing(output)
        return output
