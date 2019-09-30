import trw
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 2)
        self.fc1 = nn.Linear(20 * 7 * 7, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, batch):
        # a batch should be a dictionary of features
        x = batch['images']

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # here we create a softmax output that will use
        # the `targets` feature as classification target
        return {
            'softmax': trw.train.OutputClassification(x, 'targets')
        }


# configure and run the training/evaluation
options = trw.train.create_default_options(num_epochs=40)
trainer = trw.train.Trainer()

model, results = trainer.fit(
    options,
    inputs_fn=lambda: trw.datasets.create_cifar10_dataset(),
    run_prefix='cifar10_darts_search',
    model_fn=lambda options: Net(),
    optimizers_fn=lambda datasets, model: trw.train.create_adam_optimizers_fn(
        datasets=datasets, model=model, learning_rate=0.1))

print('DONE')
