from trw.train import OutputClassification2, OutputEmbedding
import trw
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


metrics = trw.train.metrics.default_segmentation_metrics()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_input = nn.InstanceNorm3d(1)
        self.model = trw.layers.UNetBase(3, 1, channels=[64, 128, 256], output_channels=3)

    def forward(self, batch):
        # a batch should be a dictionary of features
        x = batch['image_voxels']
        x = self.norm_input(x)
        labels = batch['label_voxels']

        # pad the input to be multiple of 8
        # TODO create a transform for this
        target_padding = 8 - np.array(x.shape[2:]) % 8
        padding = [
            0, target_padding[2],
            0, target_padding[1],
            0, target_padding[0],
        ]
        x = F.pad(x, padding)
        labels = F.pad(labels, padding)

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
            'softmax': OutputClassification2(o, labels[0], criterion_fn=trw.train.LossDiceMulticlass, metrics=metrics, collect_output=False)
        }


if __name__ == '__main__':
    datasets = trw.datasets.create_decathlon_dataset('Task04_Hippocampus')
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
            #transform_train=trw.transforms.TransformRandomCropPad([0, 2, 2, 2], mode='constant'),
            remove_patient_transform=True,
        ),
        run_prefix='decathlon_task4',
        model_fn=lambda options: Net(),
        optimizers_fn=lambda datasets, model: trw.train.create_adam_optimizers_scheduler_step_lr_fn(
            datasets=datasets,
            model=model,
            learning_rate=0.005,
            step_size=nb_epochs // 6,
            gamma=0.2
        ))

    print('DONE!')

