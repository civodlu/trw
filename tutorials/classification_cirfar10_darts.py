import trw
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, options):
        super().__init__()

        dropout_probability = options['training_parameters']['dropout_probability']
        self.convs = trw.layers.convs_2d([3, 32, 64, 96], with_batchnorm=True)
        self.denses = trw.layers.denses([1536, 256, 10], dropout_probability=dropout_probability, with_batchnorm=True, last_layer_is_output=True)

    def forward(self, batch):
        # a batch should be a dictionary of features
        x = batch['images']
        x = self.convs(x)
        x = self.denses(x)

        return {
            'softmax': trw.train.OutputClassification(x, 'targets')
        }


if __name__ == '__main__':
    # configure and run the training/evaluation
    options = trw.train.create_default_options(num_epochs=40)
    trainer = trw.train.Trainer()

    transforms = [
       trw.transforms.TransformRandomCrop(padding=[0, 2, 2])
    ]

    #transforms = None

    model, results = trainer.fit(
        options,
        inputs_fn=lambda: trw.datasets.create_cifar10_dataset(transforms=transforms, nb_workers=0),
        run_prefix='cifar10_darts_search',
        model_fn=lambda options: Net(options),
        optimizers_fn=lambda datasets, model: trw.train.create_adam_optimizers_fn(
            datasets=datasets, model=model, learning_rate=0.1))

    print('DONE')
