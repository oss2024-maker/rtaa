# -*- encoding utf-8 -*-

import enum
import importlib
import itertools
import math
import os
import pdb
import random
import shutil
import sys
import time
from turtle import forward

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from DataLoader import test_dataset_loader, test_si_dataset_loader
import Speaker_utils as util

class SpeakerNet(nn.Module):

    def __init__(self, model, trainfunc, n_out, n_class, nPerSpeaker, **kwargs):
        super(SpeakerNet, self).__init__()

        #加载模型
        SpeakerNetModel = importlib.import_module('models.' + model).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(n_out)
        
        #加载损失函数
        LossFunction = importlib.import_module('loss' + "." + trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(n_out, n_class)
        #self.__L__ = LossFunction()

        self.nPerSpeaker = nPerSpeaker


    def forward(self, data, label=None):

        #print('data before input model:',data.size()) [500, 1, 13040]->[500, 13040]
        data = data.reshape(-1, data.size()[-1]).cuda()
        #print('data after reshape:',data.size())
        #前向传播
        outp = self.__S__.forward(data)
        #print('outp after model:',outp.size())

        if label == None:
            return outp
        
        else:
            #这里关于outp的shape有点疑问？
            outp = outp.reshape(self.nPerSpeaker, -1, outp.size()[-1]).transpose(1, 0).squeeze(1)
            #计算损失
            #print('outp before loss:',outp.size())
            nloss, prec1 = self.__L__.forward(outp, label)

            return nloss, prec1


class ModelTrainer(object):

    def __init__(self, speaker_model, optimizer, scheduler, **kwargs):

        self.__model__ = speaker_model
        
        #加载优化器
        if optimizer == "Adam":
            self.__optimizer__ = util.Optimizer_Adam(self.__model__.parameters(), **kwargs)
        else:
            self.__optimizer__ = util.Optimizer_SGD(self.__model__.parameters(), optimizer, **kwargs)

        #加载学习率衰减
        self.__scheduler__, self.lr_step = util.Scheduler(self.__optimizer__, **kwargs)

        
    def train_network(self, loader, verbose):

        self.__model__.train()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss = 0
        top1 = 0
        num = 5

        tstart = time.time()

        for data, data_label in loader:
            data = data.transpose(1, 0)
            label = torch.LongTensor(data_label).cuda()
            data = data.squeeze(0).reshape(num, -1, data.size(2))
            label = label.reshape(num, -1)
            #nloss_print, prec1_print = self.__model__(data, label)
            for step in range(num):
                #每次加载一个batch
                self.__model__.zero_grad()
                data_in = data[step].unsqueeze(0)
                label_in = label[step]
                nloss, prec1 = self.__model__(data_in, label_in)
                nloss.backward()
                self.__optimizer__.step()
                loss += nloss.detach().cpu().item()

                top1 += torch.sum(prec1[0]).detach().cpu().item()
                counter += 1
            index += stepsize

            telapsed = time.time() - tstart
            tstart = time.time()

            if verbose:
                sys.stdout.write("\rProcessing {:d} of {:d}:".format(index, loader.__len__()*loader.batch_size))
                sys.stdout.write("Loss {:f} TEER/TAcc {:2.3f}% - {:.2f} Hz".format(loss/counter, top1/counter, stepsize/telapsed))
                sys.stdout.flush()
            
            if self.lr_step == 'iteration': self.__scheduler__.step()

        if self.lr_step == 'epoch': self.__scheduler__.step()
        
        return (loss/counter, top1/counter)

    def evaluateFromList(self, test_list, nDataLoaderThread, print_interval=100, num_eval=10, **kwargs):

        self.__model__.eval()
        lines = []
        files = []
        feats = {}
        tstart = time.time()

        with open(test_list) as f:
            lines = f.readlines()
        
        #测试样本list
        files = list(itertools.chain(*[x.strip().split()[-2:]  for x in lines]))
        setfiles = list(set(files))
        setfiles.sort()

        #定义测试数据加载器
        test_dataset = test_dataset_loader(setfiles, num_eval=num_eval, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size = 1,
            shuffle = False,
            num_workers = nDataLoaderThread,
            drop_last = False
        )

        #给每个测试样本提取特征
        for idx, data in enumerate(test_loader):
            #print(data[0].shape)
            #print(data[1])
            inp1 = data[0][0].cuda()
            with torch.no_grad():
                ref_feat = self.__model__(inp1).detach().cuda()
            #print(ref_feat.shape)
            feats[data[1][0]] = ref_feat
            telapsed = time.time() - tstart

            if idx % print_interval == 0:
                sys.stdout.write("\r Reading {:d} of {:d}: {:.2} Hz, embedding size {:d}".format(idx, test_loader.__len__(), idx/telapsed, ref_feat.size()[1]))
        
        all_scores = []
        all_labels = []
        all_trials = []

        tstart = time.time()

        for idx, line in enumerate(lines):
            data = line.split()
            
            feat1 = feats[data[1]].cuda()
            feat2 = feats[data[2]].cuda()

            #正则化特征
            if self.__model__.__L__.test_normalize:
                feat1 = F.normalize(feat1, p=2, dim=1)
                feat2 = F.normalize(feat2, p=2, dim=1)
            dist = F.pairwise_distance(feat1.unsqueeze(-1), feat2.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy()
            score = -1 * numpy.mean(dist)

            all_scores.append(score)
            all_labels.append(int(data[0]))
            all_trials.append(data[1] + " " + data[2])

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\r Computing {:d} of {:d}".format(idx, len(lines)))
                sys.stdout.flush()
        
        return (all_scores, all_labels, all_trials)

    def evaluateFromList_si(self, test_si_list, nDataLoaderThread, max_frames=200, **kwargs):
        self.__model__.eval()
        top1 = 0
        top10 = 0
  
        #定义测试数据加载器
        test_dataset = test_si_dataset_loader(test_si_list, max_frames, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size = 300,
            shuffle = False,
            num_workers = nDataLoaderThread,
            drop_last = False
        )


        for data, data_label in test_loader:
            # 计算top1, top10, snr, pesq
            data = data.reshape(-1, data.size()[-1]).cuda()
            #data_label = data_label.repeat(data.size(0))
            _, prec = self.__model__(data, data_label.cuda())
            top1 += torch.sum(prec[0]).detach().cpu().item()
            top10 += torch.sum(prec[1]).detach().cpu().item()
            #t_pesq += pesq

        # 计算平均输出
        sys.stdout.write("Top1: {:2.3f}%, Top10: {:2.3f}%".format(top1/len(test_loader), top10/len(test_loader)))
        sys.stdout.flush()

        print("\n")
        return 
    
    def saveParameter(self, path):
        torch.save(self.__model__.state_dict(), path)

    
    def loadParameter(self, path):
        self_state = self.__model__.state_dict()
        loaded_state = torch.load(path, map_location="cuda:0")
        for name, param in loaded_state.items():
            self_state[name].copy_(param)


            







