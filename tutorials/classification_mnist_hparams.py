import trw
import os


def create_net(hparams):
    number_hidden = hparams.create('number_hidden', trw.hparams.DiscreteIntegrer(500, 100, 1000))
    number_conv1_channels = hparams.create('number_conv1_channels', trw.hparams.DiscreteIntegrer(16, 4, 64))

    n = trw.simple_layers.Input([None, 1, 28, 28], 'images')
    n = trw.simple_layers.Conv2d(n, out_channels=number_conv1_channels, kernel_size=5, stride=2)
    n = trw.simple_layers.ReLU(n)
    n = trw.simple_layers.MaxPool2d(n, 2, 2)
    n = trw.simple_layers.Flatten(n)
    n = trw.simple_layers.Linear(n, number_hidden)
    n = trw.simple_layers.ReLU(n)
    n = trw.simple_layers.Linear(n, 10)
    n = trw.simple_layers.OutputClassification(n, output_name='softmax', classes_name='targets')
    return trw.simple_layers.compile_nn([n])
    

def evaluate_hparams(hparams):
    learning_rate = hparams.create('learning_rate', trw.hparams.ContinuousUniform(0.1, 1e-5, 1.0))

    # disable most of the reporting so that we don't end up with
    # thousands of files that are not useful for hyper-parameter
    # search
    trainer = trw.train.Trainer(
        callbacks_pre_training_fn=None,
        callbacks_post_training_fn=None,
        callbacks_per_epoch_fn=lambda: [trw.train.callback_epoch_summary.CallbackEpochSummary()])
    
    model, results = trainer.fit(
        options,
        inputs_fn=lambda: trw.datasets.create_mnist_datasset(normalize_0_1=True),
        run_prefix='run',
        model_fn=lambda options: create_net(hparams),
        optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_fn(datasets=datasets, model=model, learning_rate=learning_rate))
    
    hparam_loss = trw.train.to_value(results['outputs']['mnist']['test']['overall_loss']['loss'])
    hparam_infos = results['history']
    return hparam_loss, hparam_infos
    
    
# configure and run the training/evaluation
options = trw.train.create_default_options(num_epochs=5)
hparams_root = os.path.join(options['workflow_options']['logging_directory'], 'mnist_cnn_hparams')
trw.train.utilities.create_or_recreate_folder(hparams_root)
options['workflow_options']['logging_directory'] = hparams_root

# run the hyper-parameter search
random_search = trw.hparams.HyperParametersOptimizerRandomSearchLocal(
    evaluate_hparams_fn=evaluate_hparams,
    repeat=40)
random_search.optimize(hparams_root)

# finally analyse the run
hparams_report = os.path.join(hparams_root, 'report')
trw.hparams.analyse_hyperparameters(
    hprams_path_pattern=hparams_root + '\hparams-*.pkl',
    output_path=hparams_report)
