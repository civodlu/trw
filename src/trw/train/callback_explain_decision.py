import os
from . import callback
from . import trainer
from . import utils
from . import sample_export
from . import sequence_array
from . import outputs as outputs_trw
from . import guided_back_propagation
from . import grad_cam
from enum import Enum
import torch
import torch.nn
import numpy as np
import logging
import collections


logger = logging.getLogger(__name__)


class ExplainableAlgorithm(Enum):
    GuidedBackPropagation = 1
    GradCAM = 2


def fill_class_name(output, class_index, datasets_infos, dataset_name, split_name):
    """
    Get the class name if available, if not the class index
    """

    c_name = None
    if isinstance(output, outputs_trw.OutputClassification):
        c_names = utils.get_classification_mapping(datasets_infos, dataset_name, split_name, output.classes_name)
        if c_names is not None:
            c_names = c_names['mappinginv']
            c_name = c_names.get(class_index)
    if c_name is None:
        c_name = class_index
    return c_name


class CallbackExplainDecision(callback.Callback):
    """
    Explain the decision of a model
    """
    def __init__(self, max_samples=10, dirname='explained', dataset_name=None, split_name='test', algorithm=(ExplainableAlgorithm.GradCAM, ExplainableAlgorithm.GuidedBackPropagation), output_name=None, nb_explanations=1):
        """
        Args:
            max_samples: the maximum number of examples to export
            dirname: folder name where to export the explanations
            dataset_name: the name of the dataset to export. If `None`, the first dataset is chosen
            split_name: the split name to use
            algorithm: the algorithm (`ExplainableAlgorithm`) to be used to explain the model's decision or a list of `ExplainableAlgorithm`
            output_name: the output to be used as classification target. If `None`, report the first output belonging to a `OutputClassification`
            nb_explanations: the number of alternative explanations to be exported. nb_explanations = 1, explain the current guess, nb_explanations = 2,
                in addition calculate the explanation for the next best guess and so on for nb_explanations = 3
        """
        self.max_samples = max_samples
        self.dirname = dirname
        self.dataset_name = dataset_name
        self.split_name = split_name

        # since all `explanation` algorithms all have drawbacks, we will
        # want to export multiple explanations and so make it a list
        if isinstance(algorithm, ExplainableAlgorithm):
            self.algorithms = [algorithm]
        else:
            assert isinstance(algorithm, collections.Sequence)
            self.algorithms = algorithm

        self.batch = None
        self.output_name = output_name
        self.nb_explanations = nb_explanations

    def first_time(self, datasets):
        if self.dataset_name is None:
            self.dataset_name = next(iter(datasets))

        if datasets[self.dataset_name].get(self.split_name) is None:
            logger.error('can\'t find split={} for dataset={}'.format(self.dataset_name, self.split_name))
            self.dataset_name = None
            return

        # record a particular batch of data so that we can have the exact same samples over multiple epochs
        self.batch = next(iter(datasets[self.dataset_name][self.split_name]))

    @staticmethod
    def find_output_name(outputs, name):
        if name is not None:
            return name

        for output_name, output in outputs.items():
            if isinstance(output, outputs_trw.OutputClassification):
                return output_name

        return None

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        logger.info('started CallbackExplainDecision.__call__')
        device = options['workflow_options']['device']
        model.eval()  # we hare in evaluation mode!

        if self.batch is None:
            self.first_time(datasets)
        if self.batch is None:
            return

        root = os.path.join(options['workflow_options']['current_logging_directory'], self.dirname)
        if not os.path.exists(root):
            utils.create_or_recreate_folder(root)

        batch = utils.transfer_batch_to_device(self.batch, device=device)
        batch_size = utils.len_batch(batch)
        trainer.postprocess_batch(self.dataset_name, self.split_name, batch, callbacks_per_batch)

        outputs = model(batch)
        output_name = CallbackExplainDecision.find_output_name(outputs, self.output_name)
        if output_name is None:
            logger.error('can\'t find a classification output')
            return

        nb_samples = min(self.max_samples, utils.len_batch(self.batch))
        for algorithm in self.algorithms:
            if algorithm == ExplainableAlgorithm.GradCAM:
                # do sample by sample to simplify the export procedure
                for n in range(nb_samples):
                    batch_n = sequence_array.SequenceArray.get(
                        self.batch,
                        utils.len_batch(self.batch),
                        np.asarray([n]),
                        transforms=None,
                        use_advanced_indexing=True)

                    for tensor in batch_n.values():
                        if isinstance(tensor, torch.Tensor) and torch.is_floating_point(tensor):
                            # we want to back-propagate up to the inputs
                            tensor.requires_grad = True

                    gradcam = grad_cam.GradCam()
                    r = gradcam.generate_cam(model, batch_n, target_class_name=output_name)
                    if r is None:
                        # the algorithm failed, go to the next one
                        continue

                    selected_output_name, cams_dict = r
                    assert self.nb_explanations == 1, 'TODO handle for multiple explanations!'

                    outputs = model(batch_n)
                    output = outputs.get(selected_output_name)
                    assert output is not None
                    output_np = utils.to_value(output.output)[0]
                    max_class_indices = (-output_np).argsort()[0:self.nb_explanations]

                    for input_name, cams in cams_dict.items():
                        assert len(cams) == 1
                        enumerate_i = 0
                        c = max_class_indices[enumerate_i]  # the class output
                        c_name = fill_class_name(output, c, datasets_infos, self.dataset_name, self.split_name)

                        filename = 'sample-{}-output-{}-epoch-{}-rank-{}-alg-{}-explanation_for-{}'.format(n, input_name, len(history), enumerate_i, algorithm, c_name)
                        filename = utils.safe_filename(filename)
                        export_path = os.path.join(root, filename)

                        with open(export_path + '.txt', 'w') as f:
                            batch_n['explanation'] = cams[0].reshape([1, 1] + list(cams[0].shape))  # add a sample and a filter components
                            batch_n['output_found'] = str(output_np)
                            batch_n['output_name_found'] = c_name
                            sample_export.export_sample(batch_n, 0, export_path + '-', f)

            if algorithm == ExplainableAlgorithm.GuidedBackPropagation:
                with utils.CleanAddedHooks(model) as context:
                    gbp = guided_back_propagation.GuidedBackprop(model)
                    # do sample by sample to simplify the export procedure
                    for n in range(nb_samples):
                        batch_n = sequence_array.SequenceArray.get(
                            batch,
                            batch_size,
                            np.asarray([n]),
                            transforms=None,
                            use_advanced_indexing=True)

                        for tensor in batch_n.values():
                            if isinstance(tensor, torch.Tensor) and torch.is_floating_point(tensor):
                                # we want to back-propagate up to the inputs
                                tensor.requires_grad = True

                        outputs = model(batch_n)
                        output = outputs.get(output_name)
                        assert output is not None

                        # find the outputs with maximum probability
                        output_np = utils.to_value(output.output)[0]
                        max_class_indices = (-output_np).argsort()[0:self.nb_explanations]

                        for enumerate_i, c in enumerate(max_class_indices):
                            grad_inputs = gbp.generate_gradients(batch_n, target_class_name=output_name, target_class=c)

                            # find a meaningful name for the class, else revert to the index
                            c_name = fill_class_name(output, c, datasets_infos, self.dataset_name, self.split_name)

                            # export each input
                            for input_name, g in grad_inputs.items():
                                if g is None:
                                    continue
                                assert g.shape[0] == 1  # we should hae a single sample!

                                filename = 'sample-{}-output-{}-epoch-{}-rank-{}-alg-{}-explanation_for-{}'.format(n, input_name, len(history), enumerate_i, algorithm, c_name)
                                filename = utils.safe_filename(filename)
                                export_path = os.path.join(root, filename)

                                # for reference, export the input as well
                                with open(export_path + '.txt', 'w') as f:
                                    batch_n['explanation'] = g
                                    batch_n['output_found'] = str(output_np)
                                    positive, negative = guided_back_propagation.GuidedBackprop.get_positive_negative_saliency(g)
                                    batch_n['explanation_positive'] = positive
                                    batch_n['explanation_negative'] = negative

                                    sample_export.export_sample(batch_n, 0, export_path + '-', f)
                                    f.write('gradient norm={}\n'.format(np.linalg.norm(g)))
                                    f.write('gradient average positive={}\n'.format(np.average(g[np.where(g > 0)])))
                                    f.write('gradient average negative={}\n'.format(np.average(g[np.where(g < 0)])))

                if context.nb_hooks_removed == 0:
                    logger.error('The model doesn\'t have any ReLu node!')

        logger.info('successfully completed CallbackExplainDecision.__call__!')
