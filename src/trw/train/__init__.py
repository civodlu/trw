from .options import create_default_options
from .utilities import create_or_recreate_folder, set_optimizer_learning_rate, \
    time_it, CleanAddedHooks, safe_filename, \
    get_device, transfer_batch_to_device, find_default_dataset_and_split_names, get_class_name,\
    get_classification_mapping, get_classification_mappings, make_triplet_indices, make_pair_indices, \
    make_unique_colors, make_unique_colors_f, apply_spectral_norm, apply_gradient_clipping
from trw.train.collate import collate_dicts, collate_list_of_dicts, default_collate_fn

from .outputs_trw import Output, OutputClassification, OutputClassification2, OutputRegression, OutputEmbedding, \
    default_sample_uid_name, segmentation_criteria_ce_dice, OutputTriplets, OutputLoss, OutputSegmentation2
from .losses import LossDiceMulticlass, LossFocalMulticlass, LossTriplets, LossCenter, LossContrastive, \
    total_variation_norm, LossCrossEntropyCsiMulticlass, LossBinaryF1, one_hot, LossMsePacked
from .trainer import Trainer, create_losses_fn, epoch_train_eval, eval_loop, train_loop, \
    run_trainer_repeat, default_post_training_callbacks, default_per_epoch_callbacks, default_pre_training_callbacks, \
    default_sum_all_losses
from .trainer_across_datasets import epoch_train_eval_across_datasets
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
from .callback_skip_epoch import CallbackSkipEpoch
from .callback_save_last_model import CallbackSaveLastModel, ModelWithLowestMetric
from .callback_export_history import CallbackExportHistory
from .callback_export_classification_report import CallbackExportClassificationReport
from .callback_skip_epoch import CallbackSkipEpoch
from .callback_tensorboard import CallbackClearTensorboardLog
from .callback_tensorboard_embedding import CallbackTensorboardEmbedding
from .callback_tensorboard_record_history import CallbackTensorboardRecordHistory
from .callback_tensorboard_record_model import CallbackTensorboardRecordModel
from .callback_learning_rate_finder import CallbackLearningRateFinder
from .callback_learning_rate_recorder import CallbackLearningRateRecorder
from .callback_explain_decision import CallbackExplainDecision,  ExplainableAlgorithm
from .callback_worst_samples_by_epoch import CallbackWorstSamplesByEpoch
from .callback_zip_sources import CallbackZipSources
from .callback_export_convolution_kernel import CallbackExportConvolutionKernel

from .callback_reporting_export_samples import CallbackReportingExportSamples
from .callback_reporting_start_server import CallbackReportingStartServer
from .callback_reporting_classification_errors import CallbackReportingClassificationErrors
from .callback_reporting_model_summary import CallbackReportingModelSummary
from .callback_reporting_epoch_summary import CallbackReportingRecordHistory
from .callback_reporting_layer_statistics import CallbackReportingLayerStatistics
from .callback_reporting_dataset_summary import CallbackReportingDatasetSummary
from .callback_reporting_best_metrics import CallbackReportingBestMetrics
from .callback_reporting_augmentations import CallbackReportingAugmentations
from .callback_reporting_layer_weights import CallbackReportingLayerWeights
from .callback_debug_processes import CallbackDebugProcesses

from .sequence import Sequence
from .sequence_map import SequenceMap
from trw.train.job_executor2 import JobExecutor2
from .sequence_array import SequenceArray
from .sequence_batch import SequenceBatch
from .sequence_async_reservoir import SequenceAsyncReservoir
from .sequence_adaptor import SequenceAdaptorTorch
from .sequence_collate import SequenceCollate
from .sequence_rebatch import SequenceReBatch
from .sequence_sub_batch import SequenceSubBatch
from .sequence_array_fixed_samples_per_epoch import SequenceArrayFixedSamplesPerEpoch

from .metrics import Metric, MetricClassificationError, MetricClassificationBinarySensitivitySpecificity, MetricLoss, \
    MetricClassificationBinaryAUC, MetricClassificationF1

from .sampler import SamplerRandom, SamplerSequential, SamplerSubsetRandom, SamplerClassResampling, Sampler
from .sample_export import as_rgb_image, as_image_ui8, export_image
from trw.utils import upsample
from .filter_gaussian import FilterFixed, FilterGaussian
from .meaningful_perturbation import MeaningfulPerturbation, default_information_removal_smoothing
from .data_parallel_extented import DataParallelExtended

from .compatibility import grid_sample, affine_grid