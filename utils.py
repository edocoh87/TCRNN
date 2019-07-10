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

