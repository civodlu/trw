from typing import Optional, Any
import os
import torch
import logging


logger = logging.getLogger(__name__)


class TrainingParameters:
    """
    Define here specific training parameters
    """
    def __init__(self, num_epochs: int, mixed_precision_enabled: bool = False):
        self.num_epochs = num_epochs

        self.gradient_scaler = None
        if mixed_precision_enabled:
            try:
                from torch.cuda.amp import GradScaler
                self.gradient_scaler = GradScaler()

            except Exception as e:
                # mixed precision is not enabled
                logger.error(f'Mixed precision not enabled! Exception={e}')


class WorkflowOptions:
    """
    Define here workflow options
    """
    def __init__(self, logging_directory: str, device: torch.device):
        self.device: torch.device = device
        self.train_split: str = 'train'
        self.logging_directory: str = logging_directory
        self.current_logging_directory: str = logging_directory
        self.trainer_run: int = 0
        self.sql_database_view_path: Optional[str] = None
        self.sql_database_path: Optional[str] = None
        self.sql_database: Optional[Any] = None


class Runtime:
    """
    Define here the runtime configuration
    """
    def __init__(self):
        pass


class Options:
    """
    Create default options for the training and evaluation process.
    """
    def __init__(self,
                 logging_directory: Optional[str] = None,
                 num_epochs: int = 50,
                 device: Optional[torch.device] = None,
                 mixed_precision_enabled: bool = False
                 ):
        """

        Args:
            logging_directory: the base directory where the logs will be exported for each trained model.
                If None and if the environment variable `LOGGING_DIRECTORY` exists, it will be used as root
                directory. Else a default folder will be used

            num_epochs: the number of epochs
            device: the device to train the model on. If `None`, we will try first any available GPU then
                revert to CPU
            mixed_precision_enabled: if `True`, enable mixed precision for the training
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

        self.training_parameters: TrainingParameters = TrainingParameters(
            num_epochs=num_epochs,
            mixed_precision_enabled=mixed_precision_enabled
        )
        self.workflow_options: WorkflowOptions = WorkflowOptions(
            logging_directory=logging_directory,
            device=device
        )
        self.runtime: Runtime = Runtime()
