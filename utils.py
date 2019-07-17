from sklearn.metrics import confusion_matrix, roc_curve
from math import log10
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from pdb import set_trace as trace
import os


def print_confusion_matrix(true_labels, pred_labels):
    print('Confusion matrix:')
    mat = confusion_matrix(true_labels, pred_labels)
    format_num = lambda x: str(x) + (8 if x == 0 else 8 - int(log10(x))) * ' '
    print('    0        1\n0   {}{}\n1   {}{}'.format(format_num(mat[0, 0]), mat[0, 1],
                                                      format_num(mat[1, 0]), mat[1, 1]))


def print_normed_confusion_matrix(true_labels, pred_labels):
    print('Normalized confusion matrix:')
    mat = confusion_matrix(true_labels, pred_labels)
    norm_mat = mat / mat.sum(axis=1)[:, np.newaxis]
    print('    0        1\n0   {}  {}\n1   {}  {}'.format(str(norm_mat[0, 0])[:6], str(norm_mat[0, 1])[:6],
                                                          str(norm_mat[1, 0])[:6], str(norm_mat[1, 1])[:6]))


def plot_roc_curve(true_labels, pred_scores, save_path):
    fpr, tpr, thresholds = roc_curve(true_labels, pred_scores[:, 0], pos_label=0)
    plt.plot(fpr, tpr, 'b')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(os.path.join(save_path, 'roc_curve.png'))


def print_confusion_matrix_w_thresh(true_labels, pred_scores, thresh, thresh_type='fpr'):
    assert thresh_type in ['fpr', 'fnr']
    fpr, tpr, thresholds = roc_curve(true_labels, pred_scores[:, 0], pos_label=0)
    if thresh_type == 'fpr':
        confidence_thresh = thresholds[np.searchsorted(fpr, thresh)]
        pred_labels = (pred_scores[:, 0] <= confidence_thresh).astype(int)
        print('Normalized confusion matrix with fpr rate = {}'.format(thresh))
        print_normed_confusion_matrix(true_labels, pred_labels)
    else:  # 'fnr'
        confidence_thresh = thresholds[np.searchsorted(tpr, 1 - thresh)]
        pred_labels = (pred_scores[:, 0] <= confidence_thresh).astype(int)
        print('Normalized confusion matrix with fnr rate = {}'.format(thresh))
        print_normed_confusion_matrix(true_labels, pred_labels)


def preprocess_batch(df, n_features, take_last_k_cycles=-1):
    df = (df.drop(columns=['PC', 'DUT', 'Bank', 'BLK', 'WL', 'Str'])
          .fillna(0)
          )
    n_cycles = int(len(df.columns) / n_features)
    batch_size = len(df)
    prog_status = np.array([df['Prog_Status_cyc_{}'.format(i)].values for i in range(1, n_cycles + 1)])
    first_fail_cycle = np.argmax(prog_status, axis=0)
    labels = (first_fail_cycle > 0).astype(np.int32)
    last_input_cycle = np.random.randint(low=1 if take_last_k_cycles == -1 else take_last_k_cycles, high=n_cycles,
                                         size=labels.shape[0])
    last_input_cycle = np.where(labels == 0, last_input_cycle, first_fail_cycle)
    if take_last_k_cycles == -1:
        data = df.values.reshape(batch_size, n_cycles, n_features)
    else:
        data_arr = []
        for i in range(batch_size):
            data_arr.append(df.iloc[i].values[(last_input_cycle[i] - take_last_k_cycles) * n_features:
                                              last_input_cycle[i] * n_features])
        data = np.stack([x.reshape(take_last_k_cycles, n_features) for x in data_arr])
        last_input_cycle = [take_last_k_cycles] * batch_size
    labels = np.stack([(labels == 0).astype(np.int32), (labels == 1).astype(np.int32)]).transpose()
    return data, labels, last_input_cycle

def filter_short_strings(data, n_features, take_last_k_cycles):
    # filter out strings that fail on cycle < take_last_k_cycles
    n_cycles = int(len(data.columns) / n_features)
    prog_status = np.array(
        [data['Prog_Status_cyc_{}'.format(i)].values for i in range(1, n_cycles + 1)])
    first_fail_cycle = np.argmax(prog_status, axis=0)
    labels = (first_fail_cycle > 0).astype(np.int32)
    data = data[(labels == 0) | (first_fail_cycle > take_last_k_cycles)]
    return data
