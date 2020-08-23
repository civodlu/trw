import collections

import trw.train
import trw.datasets
import torch
import torch.nn as nn
import functools
from torch.nn import init
from trw.layers.gan import Gan, GanDataPool
from trw.train import OutputEmbedding, OutputLoss, LossMsePacked
from trw.train.outputs_trw import OutputClassification2


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class Generator2(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        #self.unet = trw.layers.UNetBase(2, input_channels=3, output_channels=3, channels=[64, 128, 256, 512])
        #self.unet = UnetGenerator(3, 3, 6, 64, use_dropout=True)
        self.unet = UnetGenerator(3, 3, 6, 96, use_dropout=True)

    def forward(self, batch, latent):
        segmentation = get_segmentation(batch)['segmentation']
        image = get_image(batch)

        o = self.unet(segmentation)
        o = torch.tanh(o)  # force -1..1 range

        l1 = torch.nn.L1Loss(reduction='none')(o, image)
        l1 = 10.0 * trw.utils.flatten(l1).mean(dim=1)
        return o, collections.OrderedDict([
            ('image', OutputEmbedding(o)),
            ('l1', OutputLoss(l1))  # L1 loss
        ])


class Discriminator2(nn.Module):
    def __init__(self):
        super().__init__()

        factor = 3
        base_filters = [64, 128, 256, 512, 512, 2]
        filters = [f * factor for f in base_filters]
        self.convs = trw.layers.convs_2d(
            6,
            filters,
            convolution_kernels=[4, 4, 4, 4, 4, 1],
            strides=[2, 2, 2, 2, 1, 1],
            batch_norm_kwargs={},
            pooling_size=None,
            padding=0,
            dropout_probability=0.2,
            activation=functools.partial(nn.LeakyReLU, negative_slope=0.2),
            last_layer_is_output=True,
            bias=False
        )

    def forward(self, batch, image, is_real):
        segmentation = get_segmentation(batch)['segmentation']

        # introduce the target as one hot encoding input to the discriminator
        o = self.convs(torch.cat([image, segmentation], dim=1))
        #o_expected = int(is_real) * torch.ones(len(image), device=image.device, dtype=torch.long)
        o_expected = torch.zeros(len(image), device=image.device, dtype=torch.float32)
        if is_real:
            # one sided label smoothing
            o_expected.uniform_(0.8, 1.2)

        o_expected = int(is_real) * torch.ones(len(image), device=image.device, dtype=torch.long)
        o_expected = o_expected.unsqueeze(1).unsqueeze(1)
        o_expected = o_expected.repeat([1, o.shape[2], o.shape[3]])
        return {
            'classification': OutputClassification2(
                o, o_expected,
                criterion_fn=LossMsePacked,  # LSGan loss function
            )
        }


def get_image(batch, source=None):
    return 2 * batch['images'] - 1


def get_segmentation(batch, source=None):
    return {
        'segmentation': 2 * batch['segmentations'] - 1
    }


def optimizer_fn(params, lr):
    optimizer = torch.optim.Adam(lr=lr, betas=(0.5, 0.999), params=params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    return optimizer, scheduler


def create_model(options):
    latent_size = 0

    discriminator = Discriminator2()
    generator = Generator2(latent_size)
    lr_base = 0.0002

    model = Gan(
        discriminator=discriminator,
        generator=generator,
        latent_size=latent_size,
        optimizer_discriminator_fn=functools.partial(optimizer_fn, lr=lr_base),  # TTUR
        optimizer_generator_fn=functools.partial(optimizer_fn, lr=lr_base / 2.0),
        real_image_from_batch_fn=get_image,
        image_pool=GanDataPool(100)
    )

    model.apply(init_weights)
    return model


def per_epoch_callbacks():
    return [
        trw.train.CallbackSkipEpoch(4, [
            trw.train.CallbackReportingExportSamples(),
        ], include_epoch_zero=True),
        trw.train.CallbackEpochSummary(),
        trw.train.CallbackReportingRecordHistory(),
    ]


def pre_training_callbacks():
    return [
        trw.train.CallbackReportingStartServer(),
    ]


options = trw.train.create_default_options(num_epochs=600, device='cuda:0')

trainer = trw.train.Trainer(
    callbacks_per_epoch_fn=per_epoch_callbacks,
)

model, result = trw.train.run_trainer_repeat(
    trainer,
    options,
    number_of_training_runs=10,
    inputs_fn=lambda: trw.datasets.create_facades_dataset(
        batch_size=10,
        transforms_train=trw.transforms.TransformCompose([
            trw.transforms.TransformRandomFlip(axis=3),
            trw.transforms.TransformRandomCropResize([192, 192], resize_mode='none')
        ])),
    eval_every_X_epoch=20,
    model_fn=create_model,
    run_prefix='facade_pix2pix',
    optimizers_fn=None  # the module has its own optimizers
)

print('DONE')