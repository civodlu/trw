from . import callback
from . import utils
from . import analysis_plots
from . import outputs as trf_outputs
import os
import matplotlib.pyplot as plt
import logging


def get_mappinginv(datasets_infos, dataset_name, split_name, class_name):
    """
    Extract the `mappinginv` from the datasets_infos
    """
    mapping = utils.get_classification_mapping(datasets_infos, dataset_name, split_name, class_name)
    if mapping is not None:
        return mapping.get('mappinginv')
    return None


logger = logging.getLogger(__name__)


class CallbackExportClassificationReport(callback.Callback):
    """
    Export the main classification measures for the classification outputs of the model

    This include:
    * text report (e.g., accuracy, sensitivity, specificity, F1, typical errors & confusion matrix)
    * confusion matrix plot
    * ROC & AUC for binary classification problems
    """
    max_class_names = 40

    def __init__(self, with_confusion_matrix=True, with_ROC=True, with_history=True, with_report=True):
        self.with_confusion_matrix = with_confusion_matrix
        self.with_ROC = with_ROC
        self.with_history = with_history
        self.with_report = with_report

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        logger.info('CallbackExportClassificationReport.__call__')
        for dataset_name, dataset in outputs.items():
            for split_name, split in dataset.items():
                for output_name, output in split.items():
                    ref = output.get('output_ref')
                    if not isinstance(ref, trf_outputs.OutputClassification):
                        continue
                    logger.info('output_classification found={}, dataset={}, split={}'.format(output_name, dataset_name, split_name))

                    root = options['workflow_options']['current_logging_directory']

                    class_name = output['output_ref'].classes_name
                    mapping = get_mappinginv(datasets_infos, dataset_name, split_name, class_name)

                    # we have a classification task, so extract important statistics & figures
                    if self.with_confusion_matrix:
                        logger.info('exporting confusion matrix')
                        title = '{}-{}-{}-cm'.format(output_name, dataset_name, split_name)

                        list_classes = None
                        if mapping is not None and len(mapping) < CallbackExportClassificationReport.max_class_names:
                            list_classes = analysis_plots.list_classes_from_mapping(mapping)

                        analysis_plots.confusion_matrix(
                            export_path=root,
                            classes_predictions=output['output'],
                            classes_trues=output['output_truth'],
                            classes=list_classes,
                            normalize=True,
                            display_numbers=False,
                            rotate_x=45,
                            title=title,
                            sort_by_decreasing_sample_size=False,  # to make the comparsion easier between splits, do not sort!
                            cmap=plt.cm.Greens
                        )

                    if self.with_ROC:
                        if output['output_raw'].shape[1] == 2:
                            logger.info('exporting ROC curve')
                            # classical ROC is only valid for binary classification
                            title = '{}-{}-{}-ROC'.format(output_name, dataset_name, split_name)

                            analysis_plots.plot_roc(
                                export_path=root,
                                trues=output['output_truth'],
                                found_scores_1=output['output_raw'][:, 1],
                                title=title,
                            )

                    if self.with_report:
                        logger.info('exporting classification report')
                        prediction_scores = output['output_raw']

                        report = analysis_plots.classification_report(
                            prediction_scores=prediction_scores,
                            trues=output['output_truth'],
                            class_mapping=mapping
                        )
                        report_name = '{}-{}-{}-report.txt'.format(output_name, dataset_name, split_name)

                        report_path = os.path.join(root, report_name)
                        with open(report_path, 'w') as f:
                            for name, value in report.items():
                                f.write('{}\n{}\n'.format(name, value))

        logger.info('CallbackExportClassificationReport.__call__ done!')
