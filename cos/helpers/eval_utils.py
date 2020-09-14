import itertools
import math

import numpy as np

from mir_eval.separation import bss_eval_sources

from cos.helpers.utils import angular_distance

def si_sdr(estimated_signal, reference_signals, scaling=True):
    """
    This is a scale invariant SDR. See https://arxiv.org/pdf/1811.02508.pdf
    or https://github.com/sigsep/bsseval/issues/3 for the motivation and
    explanation

    Input:
        estimated_signal and reference signals are (N,) numpy arrays

    Returns: SI-SDR as scalar
    """
    Rss = np.dot(reference_signals, reference_signals)
    this_s = reference_signals

    if scaling:
        # get the scaling factor for clean sources
        a = np.dot(this_s, estimated_signal) / Rss
    else:
        a = 1

    e_true = a * this_s
    e_res = estimated_signal - e_true

    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()

    SDR = 10 * math.log10(Sss/Snn)

    return SDR


def compute_sdr(gt, output, single_channel=False):
    assert(gt.shape == output.shape)
    per_channel_sdr = []

    channels = [0] if single_channel else range(gt.shape[0])
    for channel_idx in channels:
        # sdr, _, _, _ = bss_eval_sources(gt[channel_idx], output[channel_idx])
        sdr = si_sdr(output[channel_idx], gt[channel_idx])
        per_channel_sdr.append(sdr)

    return np.array(per_channel_sdr).mean()



def find_best_permutation_prec_recall(gt, output, acceptable_window=np.pi / 18):
    """
    Finds the best permutation for evaluation.
    Then uses that to find the precision and recall
    
    Inputs:
        gt, output: list of sources. lengths may differ

    Returns: Permutation that matches outputs to gt along with tp, fn and fp
    """
    n = max(len(gt), len(output))
        
    if len(gt) > len(output):
        output += [np.inf] * (n - len(output))
    elif len(output) > len(gt):
        gt += [np.inf] * (n - len(gt))

    best_perm = None
    best_inliers = -1
    for perm in itertools.permutations(range(n)):
        curr_inliers = 0
        for idx1, idx2  in enumerate(perm):
            if angular_distance(gt[idx1], output[idx2]) < acceptable_window:
                curr_inliers += 1

        if curr_inliers > best_inliers:
            best_inliers = curr_inliers
            best_perm = list(perm)

    return localization_precision_recall(best_perm, gt, output, acceptable_window)


def localization_precision_recall(permutation, gt, output, acceptable_window=np.pi/18):
    tp, fn, fp = 0, 0, 0
    for idx1, idx2 in enumerate(permutation):
        if angular_distance(gt[idx1], output[idx2]) < acceptable_window:
            tp += 1
        elif gt[idx1] == np.inf:
            fp += 1
        elif output[idx2] == np.inf:
            fn += 1
        else:
            fn += 1
            fp += 1

    return permutation, (tp, fn, fp)
