# -*- encoding: utf-8 -*-

import pdb
import time
from turtle import Turtle, forward

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import accuracy


class LossFunction(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        self.weight = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x, label=None):

        #x应该是三维[id,utter,feature]
        assert x.size()[1] >= 2 #每个说话人句子数量至少是两个
        
        gsize = x.size()[1] #gsize是每个说话人句子数量
        centrodis = torch.mean(x, 1) #每个说话人的均值
        stepsize = x.size()[0] #说话人数量

        cos_sim_matrix = []

        for ii in range(0, gsize):
            idx = [*range(0, gsize)]
            idx.remove(ii) #其余的句子
            exc_centroids = torch.mean(x[:, idx, :], 1) #其余句子的均值
            cos_sim_diag = F.cosine_similarity(x[:, ii, :], exc_centroids) #当前句子和其余句子均值的相似度
            cos_sim = F.cosine_similarity(x[:, ii, :].unsqueeze(-1), centrodis.unsqueeze(-1).transpose(0, 2))
            cos_sim[range(0, stepsize), range(0, stepsize)] = cos_sim_diag
            cos_sim_matrix.append(torch.clamp(cos_sim, 1e-6))

        cos_sim_matrix = torch.stack(cos_sim_matrix, dim=1)

        torch.clamp(self.weight, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.weight + self.b

        label = torch.from_numpy(numpy.asarray(range(0, stepsize))).cuda()
        nloss = self.criterion(cos_sim_matrix.view(-1, stepsize), torch.repeat_interleave(label.long(), repeats=gsize, dim=0).cuda())
        prec1 = accuracy(cos_sim_matrix.view(-1, stepsize).detach(), torch.repeat_interleave(label, repeats=gsize, dim=0).detach(), topk=(1,))[0].detach().cpu().item()
        
        return nloss, prec1
        
