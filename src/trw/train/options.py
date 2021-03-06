import os
import torch


def create_default_options(logging_directory=None, num_epochs=50, device=None):
    """
    Create default options for the training and evaluation process.

    Args:
        logging_directory: the base directory where the logs will be exported for each trained model.
            If None and if the environment variable `LOGGING_DIRECTORY` exists, it will be used as root
            directory. Else a default folder will be used

        num_epochs: the number of epochs

        device: the device to train the model on. If `None`, we will try first any available GPU then
            revert to CPU

    Returns:
        the options
    """

    if logging_directory is None:
        logging_directory = os.environ.get('TRW_LOGGING_ROOT')
    if logging_directory is None:
        logging_directory = 'c:/trw_logs/'
        
    if device is None:
        if torch.cuda.device_count() > 0:
            env_device = os.environ.get('TRW_DEVICE')
            if env_device is not None:
                device = torch.device(env_device)
            else:
                device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

    options = {
        'model_parameters': {
            'hyperparams': None,  # for the hyper-parameter search, this will be populated with the current parameters
            # here goes the model parameters
        },
        'training_parameters': {
            'dropout_probability': 0.5,
            'num_epochs': num_epochs,
        },
        'workflow_options': {
            'device': device,                        # the torch device to be used
            'train_split': 'train',                  # this is the split used for training
            'logging_directory': logging_directory,  # this is where all the tensorboard summary are written
            'trainer_run': 0,                        # this is the run number.

            'sql_database_view_path': None,  # configuration file for the reporting
            'sql_database_path': None,       # specify where the SQL database for the reporting is stored
            'sql_database': None,            # the SQL database used by the reporting
        },
        'runtime': {
            # here we can store the runtime configuration parameters
        }
    }

    options['workflow_options']['current_logging_directory'] = options['workflow_options']['logging_directory']
    return options
