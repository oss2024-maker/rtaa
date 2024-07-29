# -*- enconding: utf-8 -*-

import math
import pdb
import time

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append(r"/home/zhang/real_time_attack")
from Speaker_utils import accuracy


class LossFunction(nn.Module):

    def __init__(self, n_out, n_classes, margin=0.1, scale=30, easy_margin=False,  **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True

        self.m = margin
        self.s = scale
        self.in_feats = n_out
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_classes, n_out), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        #相乘,因为正则化了,所以得到的是夹角的余弦
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        #计算正弦
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        #角和公式,cos(theta+m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine>0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th)>0, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        #填充构成onehot
        one_hot.scatter_(1, label.view(-1, 1), 1)
        #增加对正确类的强调
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,10))

        return loss, prec1


        
