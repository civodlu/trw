import trw
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def export_scatter(embedding, output_truth, name, min_values=None, max_values=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = trw.train.make_unique_colors_f()

    if min_values is not None:
        assert max_values is not None
        ax.set_xlim(xmin=min_values[1], xmax=max_values[1])
        ax.set_ylim(ymin=min_values[0], ymax=max_values[0])

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

        # project to a 2D embedding so the features can be visualized directly
        self.fc2 = nn.Linear(500, 2)

        # remove the bias to force the CNN to learn 0-centered features
        self.fc3 = nn.Linear(2, 10, bias=False)

        # make sure the loss is part of the network so that the parameters
        # will be optimized.
        self.center_loss = trw.train.LossCenter(10, 2)

    def forward(self, batch):
        # a batch should be a dictionary of features
        x = batch['images'] / 255.0
        classes = batch['targets']

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = trw.layers.flatten(x)
        fc1 = F.relu(self.fc1(x))

        # No ReLu: so that we can have negative features
        embedding = self.fc2(fc1)
        x = self.fc3(embedding)

        center_loss = self.center_loss(embedding, classes)

        return {
            'softmax': trw.train.OutputClassification(x, 'targets'),
            'center_loss': trw.train.OutputLoss(center_loss),
            'embedding': trw.train.OutputEmbedding(embedding)
        }


# configure and run the training/evaluation
options = trw.train.create_default_options(num_epochs=200)
trainer = trw.train.Trainer()

model, results = trainer.fit(
    options,
    inputs_fn=lambda: trw.datasets.create_mnist_datasset(nb_workers=0),
    run_prefix='center_loss_mnist',
    model_fn=lambda options: Net(),
    optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_fn(
        datasets=datasets,
        model=model,
        learning_rate=0.051)
)


splits = ['train', 'test']
max_values = None
min_values = None
for split in splits:
    outputs_values = results['outputs']['mnist'][split]
    values = outputs_values['embedding']['output']
    if max_values is None:
        max_values = outputs_values['embedding']['output'].max(axis=0)
        min_values = outputs_values['embedding']['output'].min(axis=0)
        print("min:\n", min_values)
        print("max:\n", max_values)

    export_scatter(
        values,
        outputs_values['softmax']['output_truth'],
        f'{split}_scatter_no_center_loss',
        min_values=min_values,
        max_values=max_values
    )