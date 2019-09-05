import trw
import torch.nn as nn
import torch.nn.functional as F
import os


class Net(nn.Module):
    """
    Defines our model for MNIST
    """
    def __init__(self, hparams):
        super(Net, self).__init__()
        
        number_hidden = hparams.create('number_hidden', trw.hparams.DiscreteIntegrer(500, 100, 1000))
        number_conv1_channels = hparams.create('number_conv1_channels', trw.hparams.DiscreteIntegrer(20, 5, 100))

        self.number_hidden = int(number_hidden)
        self.number_conv1_channels = int(number_conv1_channels)
        
        self.conv1 = nn.Conv2d(1, self.number_conv1_channels, 5, 2)
        self.fc1 = nn.Linear(self.number_conv1_channels * 6 * 6, self.number_hidden)
        self.fc2 = nn.Linear(self.number_hidden, 10)

    def forward(self, batch):
        # a batch should be a dictionary of features
        x = batch['images'] / 255.0

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.number_conv1_channels * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # here we create a softmax output that will use
        # the `targets` feature as classification target
        return {
            'softmax': trw.train.OutputClassification(x, 'targets')
        }
    

def evaluate_hparams(hparams):
    # disable most of the reporting so that we don't end up with
    # thousands of files that are not useful for hyper-parameter
    # search
    learning_rate = hparams.create('learning_rate', trw.hparams.ContinuousUniform(0.1, 1e-5, 1.0))
    
    trainer = trw.train.Trainer(
        callbacks_pre_training_fn=None,
        callbacks_post_training_fn=None,
        callbacks_per_epoch_fn=lambda: [trw.train.callback_epoch_summary.CallbackEpochSummary()])
    
    # make sure we log the hyper-parameters
    options['model_parameters']['hyperparams'] = str(hparams)
    
    model, results = trainer.fit(
        options,
        inputs_fn=lambda: trw.datasets.create_mnist_datasset(),
        run_prefix='run',
        model_fn=lambda options: Net(hparams),
        optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_fn(datasets=datasets, model=model, learning_rate=learning_rate))
    
    hparam_loss = trw.train.to_value(results['outputs']['mnist']['test']['overall_loss']['loss'])
    hparam_infos = results['history']
    return hparam_loss, hparam_infos
    
    
# configure and run the training/evaluation
options = trw.train.create_default_options(num_epochs=5)
hparams_root = os.path.join(options['workflow_options']['logging_directory'], 'mnist_cnn_hparams')
trw.train.utils.create_or_recreate_folder(hparams_root)
options['workflow_options']['logging_directory'] = hparams_root

# run the hyper-parameter search
random_search = trw.hparams.HyperParametersOptimizerRandomSearchLocal(evaluate_hparams_fn=evaluate_hparams, repeat=40)
random_search.optimize(hparams_root)

# finally analyse the run
hparams_report = os.path.join(hparams_root, 'report')
trw.hparams.analyse_hyperparameters(hprams_path_pattern=hparams_root + '\hparams-*.pkl', output_path=hparams_report)
