import sed_eval
import utils
import pandas as pd
from sklearn.preprocessing import binarize, MultiLabelBinarizer
import sklearn.metrics as skmetrics
import numpy as np


def get_audio_tagging_df(df):
    return df.groupby('filename')['event_label'].unique().reset_index()


def audio_tagging_results(reference, estimated):
    """audio_tagging_results. Returns clip-level F1 Scores

    :param reference: The ground truth dataframe as pd.DataFrame
    :param estimated: Predicted labels by the model ( thresholded )
    """
    if "event_label" in reference.columns:
        classes = reference.event_label.dropna().unique().tolist(
        ) + estimated.event_label.dropna().unique().tolist()
        encoder = MultiLabelBinarizer().fit([classes])
        reference = get_audio_tagging_df(reference)
        estimated = get_audio_tagging_df(estimated)
        ref_labels, _ = utils.encode_labels(reference['event_label'],
                                            encoder=encoder)
        reference['event_label'] = ref_labels.tolist()
        est_labels, _ = utils.encode_labels(estimated['event_label'],
                                            encoder=encoder)
        estimated['event_label'] = est_labels.tolist()

    matching = reference.merge(estimated,
                               how='outer',
                               on="filename",
                               suffixes=["_ref", "_pred"])

    def na_values(val):
        if type(val) is np.ndarray:
            return val
        elif isinstance(val, list):
            return np.array(val)
        if pd.isna(val):
            return np.zeros(len(encoder.classes_))
        return val

    ret_df = pd.DataFrame(columns=['label', 'f1', 'precision', 'recall'])
    if not estimated.empty:
        matching['event_label_pred'] = matching.event_label_pred.apply(
            na_values)
        matching['event_label_ref'] = matching.event_label_ref.apply(na_values)

        y_true = np.vstack(matching['event_label_ref'].values)
        y_pred = np.vstack(matching['event_label_pred'].values)
        ret_df.loc[:, 'label'] = encoder.classes_
        for avg in [None, 'macro', 'micro']:
            avg_f1 = skmetrics.f1_score(y_true, y_pred, average=avg)
            avg_pre = skmetrics.precision_score(y_true, y_pred, average=avg)
            avg_rec = skmetrics.recall_score(y_true, y_pred, average=avg)
            # avg_auc = skmetrics.roc_auc_score(y_true, y_pred, average=avg)

            if avg == None:
                # Add for each label non pooled stats
                ret_df.loc[:, 'precision'] = avg_pre
                ret_df.loc[:, 'recall'] = avg_rec
                ret_df.loc[:, 'f1'] = avg_f1
                # ret_df.loc[:, 'AUC'] = avg_auc
            else:
                # Append macro and micro results in last 2 rows
                ret_df = ret_df.append(
                    {
                        'label': avg,
                        'precision': avg_pre,
                        'recall': avg_rec,
                        'f1': avg_f1,
                        # 'AUC': avg_auc
                    },
                    ignore_index=True)
    return ret_df


def get_event_list_current_file(df, fname):
    """
    Get list of events for a given filename
    :param df: pd.DataFrame, the dataframe to search on
    :param fname: the filename to extract the value from the dataframe
    :return: list of events (dictionaries) for the given filename
    """
    event_file = df[df["filename"] == fname]
    if len(event_file) == 1:
        if pd.isna(event_file["event_label"].iloc[0]):
            event_list_for_current_file = [{"filename": fname}]
        else:
            event_list_for_current_file = event_file.to_dict('records')
    else:
        event_list_for_current_file = event_file.to_dict('records')

    return event_list_for_current_file


def event_based_evaluation_df(reference,
                              estimated,
                              t_collar=0.200,
                              percentage_of_length=0.2):
    """
    Calculate EventBasedMetric given a reference and estimated dataframe
    :param reference: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
    reference events
    :param estimated: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
    estimated events to be compared with reference
    :return: sed_eval.sound_event.EventBasedMetrics with the scores
    """

    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=classes,
        t_collar=t_collar,
        percentage_of_length=percentage_of_length,
        empty_system_output_handling='zero_score')

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(
            reference, fname)
        estimated_event_list_for_current_file = get_event_list_current_file(
            estimated, fname)

        event_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )

    return event_based_metric


def segment_based_evaluation_df(reference, estimated, time_resolution=1.):
    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    segment_based_metric = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=classes, time_resolution=time_resolution)

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(
            reference, fname)
        estimated_event_list_for_current_file = get_event_list_current_file(
            estimated, fname)

        segment_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file)

    return segment_based_metric


def compute_metrics(valid_df, pred_df, time_resolution=1.):

    metric_event = event_based_evaluation_df(valid_df,
                                             pred_df,
                                             t_collar=0.200,
                                             percentage_of_length=0.2)
    metric_segment = segment_based_evaluation_df(
        valid_df, pred_df, time_resolution=time_resolution)
    return metric_event, metric_segment


def roc(y_true, y_pred, average=None):
    return skmetrics.roc_auc_score(y_true, y_pred, average=average)


def mAP(y_true, y_pred, average=None):
    return skmetrics.average_precision_score(y_true, y_pred, average=average)


def precision_recall_fscore_support(y_true, y_pred, average=None):
    return skmetrics.precision_recall_fscore_support(y_true,
                                                     y_pred,
                                                     average=average)


def tpr_fpr(y_true, y_pred):
    fpr, tpr, thresholds = skmetrics.roc_curve(y_true, y_pred)
    return fpr, tpr, thresholds


def obtain_error_rates_alt(y_true, y_pred, threshold=0.5):
    speech_frame_predictions = binarize(y_pred.reshape(-1, 1),
                                        threshold=threshold)
    tn, fp, fn, tp = skmetrics.confusion_matrix(
        y_true, speech_frame_predictions).ravel()

    p_miss = 100 * (fn / (fn + tp))
    p_fa = 100 * (fp / (fp + tn))
    return p_fa, p_miss


def confusion_matrix(y_true, y_pred):
    return skmetrics.confusion_matrix(y_true, y_pred)


def obtain_error_rates(y_true, y_pred, threshold=0.5):
    negatives = y_pred[np.where(y_true == 0)]
    positives = y_pred[np.where(y_true == 1)]
    Pfa = np.sum(negatives >= threshold) / negatives.size
    Pmiss = np.sum(positives < threshold) / positives.size
    return Pfa, Pmiss
