import torch
import trw
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trw.utils
from trw.train import apply_spectral_norm


class Net(nn.Module):
    """
    Defines our model for MNIST
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5, 2, padding=1)
        self.conv2 = nn.Conv2d(10, 20, 5, 1, padding=1)
        self.fc1 = nn.Linear(500, 400)
        self.fc2 = nn.Linear(400, 10)

    def forward(self, batch):
        # a batch should be a dictionary of features
        x = batch['images'] / 255.0

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = trw.utils.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # here we create a softmax output that will use
        # the `targets` feature as classification target
        return {
            'softmax': trw.train.OutputClassification(x, 'targets')
        }


def create_model(options):
    #return apply_spectral_norm(Net())
    return Net()


if __name__ == '__main__':
    options = trw.train.create_default_options(num_epochs=1000, device=torch.device('cuda:1'))
    #options = trw.train.create_default_options(num_epochs=100, device=torch.device('cpu'))
    trainer = trw.train.Trainer(
        callbacks_pre_training_fn=lambda: [
            trw.train.CallbackReportingStartServer(),
            trw.train.CallbackReportingModelSummary(),
            trw.train.CallbackReportingExportSamples(
                max_samples=10,
                table_name='test_samples'),
            trw.train.CallbackReportingDatasetSummary(),
            trw.train.CallbackReportingAugmentations(),
        ],

        callbacks_per_epoch_fn=lambda: [
            trw.train.CallbackEpochSummary(),
            trw.train.CallbackReportingRecordHistory(),
            #trw.train.CallbackReportingLayerStatistics(),
            #trw.train.CallbackReportingLayerWeights(),
            #trw.train.CallbackReportingBestMetrics(),
            trw.train.CallbackSkipEpoch(10, [
                trw.train.CallbackReportingClassificationErrors(max_samples=100),
                trw.train.CallbackSaveLastModel(keep_model_with_lowest_metric=
                                                trw.train.ModelWithLowestMetric(
                                                    dataset_name='mnist',
                                                    split_name='test',
                                                    output_name='softmax',
                                                    metric_name='loss'))
            ], include_epoch_zero=True)
        ],
        callbacks_post_training_fn=lambda: [
            trw.train.CallbackReportingClassificationErrors(max_samples=20)
        ])

    model, results = trainer.fit(
        options,
        inputs_fn=lambda: trw.datasets.create_mnist_dataset(),
        run_prefix='tmp',
        model_fn=lambda options: Net(),
        optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_fn(
            datasets=datasets, model=model, learning_rate=0.1, weight_decay=0.001))

    # calculate statistics of the final epoch
    output = results['outputs']['mnist']['test']['softmax']
    accuracy = float(np.sum(output['output'] == output['output_truth'])) / len(output['output_truth'])
    assert accuracy >= 0.95

    print('PROCESS DONE')