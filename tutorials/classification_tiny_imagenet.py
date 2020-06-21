import collections
import functools

import torch
import trw
import numpy as np
import torch.nn as nn
import torchvision


class SimpleNet(nn.Module):
    def __init__(self, options, conv_filters=[32, 64, 128, 256], conv_repeats=[3, 3, 3, 3], denses_filters=[2048, 200]):
        super().__init__()

        activation_conv = nn.LeakyReLU
        batch_norm_kwargs = {}
        dropout_probability = options['training_parameters']['dropout_probability']

        self.convs = trw.layers.convs_2d(
            input_channels=3,
            channels=conv_filters,
            activation=activation_conv,
            batch_norm_kwargs=batch_norm_kwargs,
            convolution_repeats=conv_repeats,
            convolution_kernels=3,
            last_layer_is_output=False,
        )

        self.denses = trw.layers.denses(
            [256 * 4 * 4] + denses_filters,
            last_layer_is_output=True,
            dropout_probability=dropout_probability)

    def forward(self, batch):
        images = batch['images'].float() / 255.0
        r = self.convs(images)
        r = self.denses(r)
        return collections.OrderedDict([
            ('classification', trw.train.OutputClassification(r, 'targets'))
        ])


class SimpleNet2(nn.Module):
    def __init__(self, options, conv_filters=[64, 128, 256, 512, 200], conv_repeats=[5, 4, 3, 3, 1]):
        super().__init__()

        activation_conv = nn.LeakyReLU
        batch_norm_kwargs = {}
        dropout_probability = options['training_parameters']['dropout_probability']

        self.convs = trw.layers.convs_2d(
            input_channels=3,
            channels=conv_filters,
            activation=activation_conv,
            batch_norm_kwargs=batch_norm_kwargs,
            convolution_repeats=conv_repeats,
            convolution_kernels=3,
            last_layer_is_output=True,
        )

        self.global_pooling = nn.AvgPool2d(kernel_size=[4, 4])

    def forward(self, batch):
        images = batch['images'].float() / 255.0
        r = self.convs(images)
        r = self.global_pooling(r).squeeze()
        assert len(r.shape) == 2
        return collections.OrderedDict([
            ('classification', trw.train.OutputClassification(r, 'targets'))
        ])


class Torchvision(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet34(num_classes=200)

    def forward(self, batch):
        images = batch['images'].float() / 255.0
        r = self.model.forward(images)
        return collections.OrderedDict([
            ('classification', trw.train.OutputClassification(r, 'targets'))
        ])


def per_epoch_additional_callbacks():
    return [
        trw.train.CallbackSkipEpoch(20, [
            trw.train.CallbackReportingExportSamples(max_samples=20, reporting_scatter_x='split_name', table_name='errors'),
            trw.train.CallbackSaveLastModel(model_name='interim', with_outputs=False),
        ], include_epoch_zero=True),
    ]


if __name__ == '__main__':
    transforms_train = trw.transforms.TransformCompose([
        trw.transforms.TransformCast(feature_names=['images'], cast_type='float'),
        trw.transforms.TransformRandomCutout(
            cutout_size=functools.partial(trw.transforms.cutout_random_size, min_size=[3, 5, 5], max_size=[3, 15, 15]),
            cutout_value_fn=trw.transforms.cutout_random_ui8_torch),
        #trw.transforms.TransformRandomCropResize(crop_size=[64 - 8, 64 - 8]),
        trw.transforms.TransformRandomCropPad(padding=[0, 4, 4]),
        trw.transforms.TransformRandomFlip(axis=3),
    ])

    transforms_valid = trw.transforms.TransformCompose([
        trw.transforms.TransformCast(feature_names=['images'], cast_type='float'),
    ])

    options = trw.train.create_default_options(num_epochs=2000, device=torch.device('cuda:1'))
    trainer = trw.train.Trainer(
        callbacks_per_epoch_fn=lambda: trw.train.default_per_epoch_callbacks(
            additional_callbacks=per_epoch_additional_callbacks())
    )
    model, results = trainer.fit(
        options,
        inputs_fn=lambda: trw.datasets.create_tiny_imagenet_dataset(
            batch_size=64,
            num_images_per_class=500,
            transforms_train=transforms_train,
            transforms_valid=transforms_valid),
        run_prefix='tiny_imagenet',
        #model_fn=lambda options: SimpleNet(options),
        model_fn=lambda options: SimpleNet2(options),
        #model_fn=lambda options: Torchvision(),

        eval_every_X_epoch=1,
        optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_scheduler_step_lr_fn(
            datasets=datasets,
            model=model,
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=5e-4,
            step_size=100,
            gamma=0.3,
            nesterov=True
        )
    )
