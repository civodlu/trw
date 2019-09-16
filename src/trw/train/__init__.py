from .options import create_default_options
from .trainer import Trainer, create_losses_fn, epoch_train_eval, eval_loop, train_loop, \
    create_sgd_optimizers_fn, create_sgd_optimizers_scheduler_step_lr_fn, create_scheduler_step_lr, \
    run_trainer_repeat, default_post_training_callbacks, default_per_epoch_callbacks, default_pre_training_callbacks, \
    create_adam_optimizers_fn, create_adam_optimizers_scheduler_step_lr_fn

from .outputs import Output, OutputClassification, OutputRegression, OutputEmbedding, OutputRecord, OutputSegmentation
from .utils import len_batch, create_or_recreate_folder, to_value, set_optimizer_learning_rate, \
    default_collate_fn, collate_dicts, collate_list_of_dicts, time_it, CleanAddedHooks, safe_filename, get_device
from .analysis_plots import plot_group_histories, confusion_matrix, classification_report, \
    list_classes_from_mapping, plot_roc, boxplots, export_figure, auroc
from .callback import Callback

from .graph_reflection import find_tensor_leaves_with_grad, find_last_forward_convolution, find_last_forward_types
from .grad_cam import GradCam
from .guided_back_propagation import GuidedBackprop

from .callback_epoch_summary import CallbackEpochSummary
from .callback_export_samples import CallbackExportSamples
from .callback_skip_epoch import CallbackSkipEpoch
from .callback_embedding_statistics import CallbackTensorboardEmbedding
from .callback_save_last_model import CallbackSaveLastModel
from .callback_export_history import CallbackExportHistory
from .callback_export_classification_report import CallbackExportClassificationReport
from .callback_export_augmentations import CallbackExportAugmentations
from .callback_data_summary import CallbackDataSummary
from .callback_model_summary import CallbackModelSummary
from .callback_skip_epoch import CallbackSkipEpoch
from .callback_tensorboard import CallbackClearTensorboardLog
from .callback_tensorboard_embedding import CallbackTensorboardEmbedding
from .callback_tensorboard_record_history import CallbackTensorboardRecordHistory
from .callback_tensorboard_record_model import CallbackTensorboardRecordModel
from .callback_export_best_history import CallbackExportBestHistory
from .callback_export_classification_errors import CallbackExportClassificationErrors
from .callback_learning_rate_finder import CallbackLearningRateFinder
from .callback_learning_rate_recorder import CallbackLearningRateRecorder
from .callback_explain_decision import CallbackExplainDecision,  ExplainableAlgorithm
from .callback_export_classification_by_epoch import CallbackExportClassificationByEpoch

from .sequence import Sequence
from .sequence_map import SequenceMap, JobExecutor
from .sequence_array import SequenceArray
from .sequence_batch import SequenceBatch
from .sequence_async_reservoir import SequenceAsyncReservoir
from .sequence_adaptor import SequenceAdaptorTorch
from .sequence_collate import SequenceCollate

from .sampler import SamplerRandom, SamplerSequential, SamplerSubsetRandom, SamplerClassResampling, Sampler

from .sample_export import as_rgb_image, as_image_ui8, export_image

from .losses import DiceLoss
