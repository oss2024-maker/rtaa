# -*- encoding: utf-8 -*-

import argparse
import datetime
import glob
import os
import pdb
import socket
import sys
import time
import warnings
import zipfile
from random import shuffle

import numpy
import torch
import yaml

from DataLoader import *
from Speaker_net import *
from Speaker_utils import *


#加载配置文件
def load_hparam(filename):
    stream = open(filename, 'r')
    docs = yaml.load_all(stream, Loader=yaml.FullLoader)
    hparam_dict = dict()
    for doc in docs:
        for k, v in doc.items():
            hparam_dict[k] = v
    return hparam_dict

class Dotdict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            #如果value仍然是字典
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value

class Hparam(Dotdict):

    def __init__(self, file):
        super(Dotdict, self).__init__()
        hp_dict = load_hparam(file)
        hp_dotdict = Dotdict(hp_dict)
        for k, v in hp_dotdict.items():
            setattr(self, k, v)
    
    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__


def main_worker(hp, path_model, path_reslut):

    #加载模型
    hp_dict = {}
    for key, value in hp.items():
        for k, v in value.items():
            hp_dict[k] = v
    #print(hp_dict)
    s = SpeakerNet(**hp_dict).cuda()

    it = 1
    eers = [100]

    scorefile = open(path_reslut + "/scores.txt", "a+")

    #初始化数据加载器
    if hp.dataload.sampler == True:
        train_dataset = train_dataset_loader(**hp_dict)
        train_sampler = train_dataset_sampler(train_dataset, **hp_dict)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = hp.dataload.batch_size,
            num_workers = hp.dataload.nDataLoaderThread,
            pin_memory = False,
            sampler = train_sampler,
            worker_init_fn = worker_init_fn,
            drop_last = True,
        )
    else:
        train_dataset = train_dataset_loader_wosampler(**hp_dict)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = hp.dataload.batch_size,
            num_workers = hp.dataload.nDataLoaderThread,
            pin_memory = False,
            shuffle = True,
            worker_init_fn = worker_init_fn,
            drop_last = True,
        )


    #加载训练器
    trainer = ModelTrainer(s, **hp_dict)
    
    # #加载模型权重
    # modelfiles = glob.glob('%s/model0*.model'%path_model)
    # modelfiles.sort()

    if hp.load_save.init_model != "":
        trainer.loadParameter(hp.load_save.init_model)
        print("model {} loaded!".format(hp.load_save.init_model))
    # elif len(modelfiles) >= 1:
    #     trainer.loadParameter(modelfiles[-1])
    #     print("model {} loaded from previous state!".format(modelfiles[-1]))
    #     it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
    
    for ii in range(1, 10):
        trainer.__scheduler__.step()
        hp.optimizer.lr
    if hp.moudle.train == False:

        pytorch_total_params = sum(p.numel() for p in s.__S__.parameters())
        print('Total parameters:', pytorch_total_params)
        print('Test list', hp.data.test_list)

        trainer.evaluateFromList_si(**hp_dict)
        sc, lab, _ = trainer.evaluateFromList(**hp_dict)
        result = tuneThresholdfromScore(sc, lab, [1,0,1])
        fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
        mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, hp.evaluation.dcf_p_target, hp.evaluation.dcf_c_miss, hp.evaluation.dcf_c_fa)

        print('\n', time.strftime("%Y-%m-%d %H:%M:%S"), "VEER {:2.4f}".format(result[1]), "MinDCF {:2.5f}".format(mindcf))
        return
    
    for it in range(it, hp.train.max_epoch + 1):
        lr = max([x['lr'] for x in trainer.__optimizer__.param_groups])
        loss, traineer = trainer.train_network(train_loader, verbose=True)
        #记录结果
        print('\n', time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch{:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f}".format(it, traineer, loss, lr))
        scorefile.write("Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f} \n".format(it, traineer, loss, lr))

        #评估
        if it % hp.train.test_interval == 0:
            sc, lab, _ = trainer.evaluateFromList(**hp_dict)
            result = tuneThresholdfromScore(sc, lab, [1,0,1])
            fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
            mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, hp.evaluation.dcf_p_target, hp.evaluation.dcf_c_miss, hp.evaluation.dcf_c_fa)

            eers.append(result[1])
            print('\n', time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, VEER {:2.4}, MinDCF {:2.5f}".format(it, result[1], mindcf))
            scorefile.write("Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}\n".format(it, result[1], mindcf))

            path = hp.load_save.save_path + "_" + hp.model.model + "_" + hp.train.trainfunc
            trainer.saveParameter(path + "/model" + "/model%09d.model"%it)
            with open(path + "/model" + "/model%09d.eer"%it, 'w') as eerfile:
                eerfile.write("{:2.4f}".format(result[1]))
            #trainer.evaluateFromList_si(**hp_dict)
            scorefile.flush()

    scorefile.close()


def main(hp):
    path = hp.load_save.save_path + "_" + hp.model.model + "_" + hp.train.trainfunc
    path_model = path + "/model"
    path_result = path + "/result"
    os.makedirs(path_model, exist_ok=True)
    os.makedirs(path_result, exist_ok=True)
    main_worker(hp, path_model, path_result)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    hp = Hparam(file='configs/config_LibriSpeech.yaml')
    main(hp)

