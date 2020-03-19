from .options import create_default_options
from .utilities import len_batch, create_or_recreate_folder, to_value, set_optimizer_learning_rate, \
    default_collate_fn, collate_dicts, collate_list_of_dicts, time_it, CleanAddedHooks, safe_filename, \
    get_device, transfer_batch_to_device, find_default_dataset_and_split_names, get_class_name,\
    get_classification_mapping, get_classification_mappings, safe_lookup, flatten_nested_dictionaries, clamp_n, \
    make_triplets, make_unique_colors, make_unique_colors_f

from .outputs import Output, OutputClassification, OutputRegression, OutputEmbedding, OutputRecord, \
    OutputSegmentation, default_sample_uid_name, segmentation_criteria_ce_dice, OutputTriplets, OutputLoss
from .trainer import Trainer, create_losses_fn, epoch_train_eval, eval_loop, train_loop, \
    run_trainer_repeat, default_post_training_callbacks, default_per_epoch_callbacks, default_pre_training_callbacks, \
    default_sum_all_losses
from .optimizers import create_sgd_optimizers_fn, create_sgd_optimizers_scheduler_step_lr_fn, \
    create_scheduler_step_lr, create_adam_optimizers_fn, \
    create_adam_optimizers_scheduler_step_lr_fn, create_optimizers_fn
from .analysis_plots import plot_group_histories, confusion_matrix, classification_report, \
    list_classes_from_mapping, plot_roc, boxplots, export_figure, auroc
from .callback import Callback
from .graph_reflection import find_tensor_leaves_with_grad, find_last_forward_convolution, \
    find_last_forward_types, find_first_forward_convolution
from .grad_cam import GradCam
from .guided_back_propagation import GuidedBackprop, post_process_output_for_gradient_attribution
from .integrated_gradients import IntegratedGradients

from .callback_epoch_summary import CallbackEpochSummary
from .callback_export_samples import CallbackExportSamples
from .callback_skip_epoch import CallbackSkipEpoch
from .callback_embedding_statistics import CallbackEmbeddingStatistics
from .callback_save_last_model import CallbackSaveLastModel
from .callback_export_history import CallbackExportHistory
from .callback_export_classification_report import CallbackExportClassificationReport
from .callback_export_augmentations import CallbackExportAugmentations
from .callback_data_summary import CallbackDataSummary
from .callback_model_summary import CallbackModelSummary, model_summary
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
from .callback_worst_samples_by_epoch import CallbackWorstSamplesByEpoch
from .callback_activation_statistics import CallbackActivationStatistics
from .callback_zip_sources import CallbackZipSources
from .callback_export_convolution_kernel import CallbackExportConvolutionKernel
from .callback_export_segmentations import CallbackExportSegmentations

from .sequence import Sequence
from .sequence_map import SequenceMap, JobExecutor
from .sequence_array import SequenceArray
from .sequence_batch import SequenceBatch
from .sequence_async_reservoir import SequenceAsyncReservoir
from .sequence_adaptor import SequenceAdaptorTorch
from .sequence_collate import SequenceCollate
from .sequence_rebatch import SequenceReBatch

from .metrics import Metric, MetricClassificationError, MetricClassificationSensitivitySpecificity, MetricLoss

from .sampler import SamplerRandom, SamplerSequential, SamplerSubsetRandom, SamplerClassResampling, Sampler
from .sample_export import as_rgb_image, as_image_ui8, export_image
from .losses import LossDiceMulticlass, LossFocalMulticlass, LossTriplets, LossCenter
from .upsample import upsample
from .filter_gaussian import FilterFixed, FilterGaussian
from .meaningful_perturbation import MeaningfulPerturbation, default_information_removal_smoothing
from .data_parallel_extented import DataParallelExtended

from .compatibility import grid_sample