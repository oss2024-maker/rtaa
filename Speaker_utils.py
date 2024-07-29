# -*- encoding: utf-8 -*-
####
import glob
import os
import pdb
import sys
import time
from operator import itemgetter

import numpy
import torch
import torch.nn.functional as F
from sklearn import metrics


def accuracy(output, target, topk=(1,)):
    #计算准确率
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    
    return res

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float=0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )
    
    def forward(self, input: torch.tensor):
        #print('input:', input.size())
        input = input.unsqueeze(1)
        #print(input.size())
        input = F.pad(input, (1, 0), 'reflect')
        
        return F.conv1d(input, self.flipped_filter).squeeze(1)

def replace_none_with_mean(lst):
    # 过滤掉None值
    filtered_lst = [x for x in lst if x is not None]
    # 计算平均值
    mean_value = sum(filtered_lst) / len(filtered_lst) if filtered_lst else None
    # 替换列表中的None值
    return [mean_value if x is None else x for x in lst]

def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
    fpr, tpr, thresholds = metrics.roc_curve(labels, replace_none_with_mean(scores), pos_label=1)
    fnr = 1 - tpr

    tunedThreshold = [];
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])*100
    
    return (tunedThreshold, eer, fpr, fnr);

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):

      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      sorted_labels = []
      labels = [labels[i] for i in sorted_indexes]
      fnrs = []
      fprs = []

      # At the end of this loop, fnrs[i] is the number of errors made by
      # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
      # is the total number of times that we have correctly accepted scores
      # greater than thresholds[i].
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i-1] + labels[i])
              fprs.append(fprs[i-1] + 1 - labels[i])
      fnrs_norm = sum(labels)
      fprs_norm = len(labels) - fnrs_norm

      # Now divide by the total number of false negative errors to
      # obtain the false positive rates across all thresholds
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      # Divide by the total number of corret positives to get the
      # true positive rate.  Subtract these quantities from 1 to
      # get the false positive rates.
      fprs = [1 - x / float(fprs_norm) for x in fprs]
      return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def

    return min_dcf, min_c_det_threshold

#学习率衰减
def Scheduler(optimizer, test_interval, lr_decay, **kwargs):

    sche_fn = torch.optim.lr_scheduler.StepLR(optimizer, step_size=test_interval, gamma=lr_decay)
    lr_step = 'epoch'

    return sche_fn, lr_step

#优化器
def Optimizer_Adam(parameters, lr, weight_decay, **kwargs):
    
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

def Optimizer_SGD(parameters, lr, weight_decay, **kwargs):

    return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)


def snr_out(signal_source, signal_noise):
#   signal_noise = signal_source - signal_source_noise
#   mean_signal_source = torch.mean(signal_source)
#   signal_source = signal_source - mean_signal_source
    signal_power = torch.sum(signal_source**2, dim=1)
    noise_power = torch.sum(signal_noise**2, dim=1)
    power_rate = torch.div(signal_power, noise_power)
    snr = 10 * torch.log10(power_rate)
    snr = torch.mean(snr)
    return snr
