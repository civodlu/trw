import functools

from trw.layers import default_layer_config, BlockConvNormActivation, NormType
from trw.train import OutputEmbedding, OutputSegmentation2
import trw
import torch.nn as nn


class UNetSegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_input = nn.InstanceNorm3d(1)
        self.model = trw.layers.UNetBase(3, 1, channels=[64, 128, 256], output_channels=3)

    def forward(self, batch):
        # a batch should be a dictionary of features
        labels = batch['label_voxels']
        x = batch['image_voxels']
        x = self.norm_input(x)

        o = self.model(x)

        x_2d = x[:, :, x.shape[2] // 2]
        x_2d = trw.transforms.resize(x_2d, (64, 64))

        o_2d = o[:, :, o.shape[2] // 2]
        o_2d = trw.transforms.resize(o_2d, (64, 64))

        labels_2d = labels[:, :, o.shape[2] // 2]
        labels_2d = trw.transforms.resize(labels_2d, (64, 64))

        return {
            'x_2d': OutputEmbedding(x_2d),
            'o_2d': OutputEmbedding(nn.Sigmoid()(o_2d)),
            'labels_2d': OutputEmbedding(labels_2d),
            'softmax': OutputSegmentation2(o, labels)
        }


class ResnetSegmentation(nn.Module):
    def __init__(self, options):
        super().__init__()
        config = default_layer_config()
        I_O = functools.partial(BlockConvNormActivation, kernel_size=7)
        self.model = trw.layers.EncoderDecoderResnet(
            3,
            input_channels=1,
            output_channels=3,
            encoding_channels=[64, 128],
            decoding_channels=[128, 64],
            init_block=I_O,
            out_block=I_O,
            config=config,
            nb_residual_blocks=18,
        )

        self.norm_input = nn.InstanceNorm3d(1)

        self.options = options

    def forward(self, batch):
        labels = batch['label_voxels']
        x = batch['image_voxels']
        x = self.norm_input(x)

        o = self.model(x)

        x_2d = x[:, :, x.shape[2] // 2]
        x_2d = trw.transforms.resize(x_2d, (64, 64))

        o_2d = o[:, :, o.shape[2] // 2]
        o_2d = trw.transforms.resize(o_2d, (64, 64))

        labels_2d = labels[:, :, o.shape[2] // 2]
        labels_2d = trw.transforms.resize(labels_2d, (64, 64))

        return {
            'x_2d': OutputEmbedding(x_2d),
            'o_2d': OutputEmbedding(nn.Sigmoid()(o_2d)),
            'labels_2d': OutputEmbedding(labels_2d),
            'softmax': OutputSegmentation2(o, labels)
        }


if __name__ == '__main__':
    nb_epochs = 400
    options = trw.train.create_default_options(num_epochs=nb_epochs)
    trainer = trw.train.Trainer(
        callbacks_pre_training_fn=lambda: [
            trw.train.CallbackReportingStartServer(),
            trw.train.CallbackReportingModelSummary(),
            trw.train.CallbackReportingDatasetSummary(),
            trw.train.CallbackReportingAugmentations(),
        ],
        callbacks_per_epoch_fn=lambda: [
            trw.train.CallbackLearningRateRecorder(),
            trw.train.CallbackEpochSummary(),
            trw.train.CallbackReportingRecordHistory(),
            trw.train.CallbackReportingBestMetrics(),
            trw.train.CallbackSkipEpoch(10, [
                trw.train.CallbackReportingExportSamples(table_name='current_samples'),
            ], include_epoch_zero=True)
        ]
    )

    model, results = trainer.fit(
        options,
        inputs_fn=lambda: trw.datasets.create_decathlon_dataset(
            'Task04_Hippocampus',
            transform_train=trw.transforms.TransformResizeModuloCropPad(multiple_of=8, mode='pad'),
            transform_valid=trw.transforms.TransformResizeModuloCropPad(multiple_of=8, mode='pad'),
            remove_patient_transform=True,
        ),
        run_prefix='decathlon_task4',
        model_fn=lambda options: UNetSegmentation(),
        #model_fn=lambda options: ResnetSegmentation(options),
        optimizers_fn=lambda datasets, model: trw.train.create_adam_optimizers_scheduler_step_lr_fn(
            datasets=datasets,
            model=model,
            #learning_rate=0.005,
            learning_rate=0.02,
            step_size=nb_epochs // 6,
            gamma=0.2
        ))

    print('DONE!')

