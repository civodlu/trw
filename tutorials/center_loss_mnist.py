import trw
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def export_scatter(embedding, output_truth, name):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = trw.train.make_unique_colors_f()

    for c in range(0, 10):
        c_indices = np.where(output_truth == c)
        x = embedding[c_indices][:, 1]
        y = embedding[c_indices][:, 0]
        ax.scatter(x, y, alpha=0.8, c=colors[c], edgecolors='none', s=10, label=str(c))

    plt.title(name)
    plt.legend(loc=2)
    trw.train.export_figure(options['workflow_options']['current_logging_directory'], name)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 2)
        self.fc1 = nn.Linear(20 * 6 * 6, 500)
        self.fc2 = nn.Linear(500, 2)  # project to a 2D embedding so the features can be visualized directly
        self.fc3 = nn.Linear(2, 10)

    def forward(self, batch):
        # a batch should be a dictionary of features
        x = batch['images'] / 255.0

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = trw.layers.flatten(x)
        x = F.relu(self.fc1(x))
        embedding = F.relu(self.fc2(x))
        x = self.fc3(embedding)

        # here we create a softmax output that will use
        # the `targets` feature as classification target
        return {
            'softmax': trw.train.OutputClassification(x, 'targets'),
            'embedding': trw.train.OutputEmbedding(embedding)
        }


# configure and run the training/evaluation
options = trw.train.create_default_options(num_epochs=40)
trainer = trw.train.Trainer()

model, results = trainer.fit(
    options,
    inputs_fn=lambda: trw.datasets.create_mnist_datasset(),
    run_prefix='center_loss_mnist',
    model_fn=lambda options: Net(),
    optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_fn(
        datasets=datasets, model=model, learning_rate=0.1))


splits = ['train', 'test']
for split in splits:
    outputs_values = results['outputs']['mnist'][split]
    export_scatter(
        outputs_values['embedding']['output'],
        outputs_values['softmax']['output_truth'],
        f'{split}_scatter_no_center_loss')