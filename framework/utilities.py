import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def calculate_scalar(x):
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)
    return mean, std


def scale(x, mean, std):
    return (x - mean) / std



def cal_acc(targets, predictions):
    tagging_truth_label_matrix = targets
    pre_tagging_label_matrix = predictions

    # overall
    tp = np.sum(pre_tagging_label_matrix + tagging_truth_label_matrix > 1.5)
    fn = np.sum(tagging_truth_label_matrix - pre_tagging_label_matrix > 0.5)
    fp = np.sum(pre_tagging_label_matrix - tagging_truth_label_matrix > 0.5)
    tn = np.sum(pre_tagging_label_matrix + tagging_truth_label_matrix < 0.5)

    Acc = (tp + tn) / (tp + tn + fp + fn)
    return Acc




