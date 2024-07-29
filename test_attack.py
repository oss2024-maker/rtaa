import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import soundfile as sf
import itertools
import math
import os
import pdb
import random
import shutil
import sys
from pesq import pesq
import time
from torch.utils.data import Dataset, DataLoader
from Speaker_net import *
from spr_utils import *
import matplotlib.pyplot as plt
import librosa
from RIR import torch_conv


#测试数据集类

#测试数据集类
class test_sv_dataset_loader(Dataset):

    def __init__(self, test_list, eval_frames, num_eval, enroll, **kwargs):
        self.max_frames = eval_frames
        self.num_eval = num_eval
        self.test_list = test_list
        self.enroll = enroll

    def __getitem__(self, index):
        audio = loadWAV(self.test_list[index], self.max_frames, evalmode=True, num_eval=self.num_eval, enroll=self.enroll)

        return torch.FloatTensor(audio), self.test_list[index]

    def __len__(self):
        
        return len(self.test_list)


class test_dataset_loader(Dataset):

    def __init__(self, test_list, eval_frames, enroll, **kwargs):
        self.max_frames = eval_frames
        self.test_list = test_list
        self.enroll = enroll
        
        #读取训练集列表
        with open(test_list) as dataset_file:
            lines = dataset_file.readlines()

        #构建一个字典,key是说话人的名字,value是说话人的序号
        self.dictkeys = list(set([x.split()[0] for x in lines]))
        self.dictkeys.sort()
        self.dictkeys = {key : ii for ii, key in enumerate(self.dictkeys)}

        #构建一个数据集路径的列表,和一个数据集标签的列表
        self.data_list = []
        self.data_label = []

        for lidx, line in enumerate(lines):
            data = line.strip().split()

            #说话人标签
            speaker_label = self.dictkeys[data[0]]
            #语音路径
            filename = data[1]

            self.data_label.append(speaker_label)
            self.data_list.append(filename)

    def __getitem__(self, indice):

        audio = loadWAV(self.data_list[indice], self.max_frames, evalmode=False, enroll=self.enroll)
        return torch.FloatTensor(audio), self.data_label[indice]

    def __len__(self):
        #可以len(class)查看数据量
        return len(self.data_list)
    
    # def __target_label__(self, target):
    #     num = self.dictkeys[target]
        # return num

def loadWAV(filename, max_frames, evalmode=False, num_eval=5, enroll=3200):
    #读取音频文件
    audio, sample_rate = sf.read(filename)
    
    # plt.figure()
    # librosa.display.waveshow(audio)
    # plt.show()
    #max_audio = max_frames * 160 + 240
    max_audio = 2 * enroll

    #如果音频长度小于最大长度则补齐
    audiosize = audio.shape[0]
    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), 'constant')
        audiosize = audio.shape[0]

    if evalmode:
        #测试时采用等距方法选取开始位置
        startframe = np.int64(np.linspace(0, audiosize-max_audio, num=num_eval))

        feats = []
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

        #把选中的音频段堆起来
        feat = np.stack(feats, axis=0).astype(np.float)

        return feat
    else:
        # 音频截取
        #start = random.randint(0, audiosize-max_audio)
        start = 0
        feat = audio[int(start):int(start)+max_audio].astype(float)

        return feat

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz).cuda()) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def assert_rank(expected_rank):
    expected_rank_dict = {}
    if isinstance(expected_rank, int):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True


def get_shape_list(tensor, expected_rank=None):
    if expected_rank is not None:
        assert_rank(expected_rank)

    shape = tensor.size()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    assert False, "Static shape not available for {}".format(tensor)

    dyn_shape = tensor.size()
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape
def rand_noise(origin_data, snr):
    perb = torch.randn(origin_data.size()).cuda()
    for index in range(perb.size()[0]):
        origin_l2 = torch.sum(origin_data[index]**2)
        power_perb = origin_l2 / (10.0**(snr/10))
        perb_l2 = torch.sum(perb[index]**2)
        perb[index] = perb[index] * torch.sqrt((power_perb / perb_l2))
        #perb_l2 = torch.sum(perb[index]**2)

    return perb


def evaluateFromList_si(s, sap, test_list, nDataLoaderThread, eval_frames, result_save_path, epoch, enroll, hidden_size, batch_size, perb_uni, snr, target, adv_attack=True, model_attack=True, **kwargs):

    top1 = 0
    top10 = 0
    dist_self = 0
    t_snr = 0
    t_pesq = 0
    p = 0

    scorefile = open(result_save_path, "a")

    #定义测试数据加载器
    test_dataset = test_dataset_loader(test_list, eval_frames=eval_frames, enroll=enroll, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = nDataLoaderThread,
        drop_last = True
    )

    #target_label = test_dataset.__target_label__(target)

    for data, data_label in test_loader:
        if model_attack == True:
            # 预测器生成扰动
            lookback = data[:, :enroll].unsqueeze(1).cuda()
            with torch.no_grad():
                perb, _ = sap(perb_uni, lookback)
                perb = perb.squeeze(1)
        else:
            # 种子扰动
            perb = perb_uni.squeeze(1)
        
        # 约束扰动asnr
        origin_l2 = torch.sum(data[:, int(0.5*enroll):int(1.5*enroll)]**2)
        power_perb = origin_l2 / (10.0**(25/10))
        perb_l2 = torch.sum(perb**2)
        perb = perb * torch.sqrt((power_perb / perb_l2))

        # # 加噪,约束nsnr
        # noise = torch.randn_like(perb)
        # power_perb = origin_l2 / (10.0**(35/10))
        # perb_l2 = torch.sum(noise**2)
        # noise = noise * torch.sqrt((power_perb / perb_l2))
        # perb = perb + noise

        # # 扰动约束fsnr
        # origin_l2 = torch.sum(data[:, int(0.5*enroll):int(1.5*enroll)]**2)
        # power_perb = origin_l2 / (10.0**(25/10))
        # perb_l2 = torch.sum(perb**2)
        # perb = perb * torch.sqrt((power_perb / perb_l2))

        # # 仅rir攻击添加
        # perb_repeat = perb.repeat(perb.size()[0], 1)
        # perb_amp = torch_conv(data[:, int(0.5*enroll):int(1.5*enroll)].cuda(), perb_repeat).squeeze(1) - data[:, int(0.5*enroll):int(1.5*enroll)].cuda()
        # origin_l2 = torch.sum(data[:, int(0.5*enroll):int(1.5*enroll)].cuda()**2)
        # power_perb = origin_l2 / (10.0**(30/10))
        # perb_l2 = torch.sum(perb_amp**2)
        # perb_repeat = perb_amp * torch.sqrt((power_perb / perb_l2))

        adv = data[:, int(0.5*enroll):int(1.5*enroll)].cuda() + perb

        adv_feat = s(adv)
        speaker_feature = s(data[:, :enroll])  # 计算与自身特征距离时用
        dist = torch.pairwise_distance(speaker_feature, adv_feat)

        snr_outs = snr_out(data[:, int(0.5*enroll):int(1.5*enroll)].cuda(), perb)
        try:
            p = pesq(16000, data[:, int(0.5*enroll):int(1.5*enroll)][0].detach().cpu().numpy(), adv[0].detach().cpu().numpy())
        except:
            p = p
        t_pesq += p

        # # 随机重采样
        # start = random.randint(0, enroll)
        # adv = adv[:, start:start+enroll]

        # 计算top1, top10, snr, pesq
        #target_label_tensor = torch.tensor(data.size(0)*[target_label])
        _, prec = s(adv, data_label.cuda())
        top1 += torch.sum(prec[0]).detach().cpu().item()
        top10 += torch.sum(prec[1]).detach().cpu().item()
        dist_self += torch.mean(dist).detach().cpu().item()
        t_snr += snr_outs.detach().cpu().item()

    # 计算平均输出
    sys.stdout.write("Epoch: {:d}, Top1: {:2.3f}%, Top10: {:2.3f}%, Dist: {:2.3f}, SNR: {:2.3f}, PESQ: {:2.3f}".format(epoch, top1/len(test_loader), top10/len(test_loader), dist_self/len(test_loader), t_snr/len(test_loader), t_pesq/len(test_loader)))
    sys.stdout.flush()

    # 写入结果
    scorefile.write("Epoch: {:d}, Top1: {:2.3f}%, Top10: {:2.3f}%, Dist: {:2.3f}, SNR: {:2.3f}, PESQ: {:2.3f}\n".format(epoch, top1/len(test_loader), top10/len(test_loader), dist_self/len(test_loader), t_snr/len(test_loader), t_pesq/len(test_loader)))
    scorefile.close()

    print("\n")
    return 

def evaluateFromList_sv(s, sap, test_list, nDataLoaderThread, eval_frames, elen, result_save_path, hidden_size, dcf_p_target, dcf_c_miss, dcf_c_fa, print_interval=100, num_eval=10, **kwargs):

    lines = []
    files = []
    feats_ori = {}
    feats_adv = {}
    tstart = time.time()
    scorefile = open(result_save_path, "a")

    with open(test_list) as f:
        lines = f.readlines()
    
    #测试样本list
    files = list(itertools.chain(*[x.strip().split()[-2:]  for x in lines]))
    setfiles = list(set(files))
    setfiles.sort()

    #定义测试数据加载器
    test_dataset = test_sv_dataset_loader(setfiles, num_eval=num_eval, eval_frames=eval_frames, enroll=elen, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = nDataLoaderThread,
        drop_last = False
    )

    #给每个测试样本提取特征
    adv1 = torch.zeros(num_eval, elen).cuda()
    adv1 = adv1.reshape(adv1.size()[0], int(adv1.size()[1]/hidden_size), hidden_size)
    t_attn_mask = generate_square_subsequent_mask(adv1.size()[1])

    for idx, data in enumerate(test_loader):
        #print(data[0].shape)
        #print(data[1])
        inp = data[0].cuda()
        inp = inp.permute(1, 0, 2)
        with torch.no_grad():
            inp_ori = inp[:, :, :elen]
            ref_feat_ori = s(inp_ori.squeeze(1)).detach().cuda()

            inp1 = inp[:, :, :elen]
            inp2 = inp[:, :, elen:2*elen]
            perb1 = sap(inp1).squeeze(1)

            adv2 = inp2.squeeze(1) + perb1
            ref_feat_adv = s(adv2).detach().cuda()

        feats_ori[data[1][0]] = ref_feat_ori
        feats_adv[data[1][0]] = ref_feat_adv
        telapsed = time.time() - tstart

        if idx % print_interval == 0:
            sys.stdout.write("\r Reading {:d} of {:d}: {:.2} Hz, embedding size {:d}".format(idx, test_loader.__len__(), idx/telapsed, ref_feat_ori.size()[1]))

    print("\n")
    all_scores = []
    all_labels = []
    all_trials = []

    tstart = time.time()

    for idx, line in enumerate(lines):
        data = line.split()
        
        feat1 = feats_adv[data[1]].cuda()
        feat2 = feats_ori[data[2]].cuda()

        #正则化特征
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)
        #dist = F.pairwise_distance(feat1.unsqueeze(-1), feat2.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy()
        dist = torch.pairwise_distance(feat1, feat2)
        dist = torch.mean(dist).detach().cpu().numpy()
        score = -1 * dist

        all_scores.append(score)
        all_labels.append(int(data[0]))
        all_trials.append(data[1] + " " + data[2])

        if idx % print_interval == 0:
            telapsed = time.time() - tstart
            sys.stdout.write("\r Computing {:d} of {:d}".format(idx, len(lines)))
            sys.stdout.flush()

    # 计算平均输出
    result = tuneThresholdfromScore(all_scores, all_labels, [1,0,1])
    fnrs, fprs, thresholds = ComputeErrorRates(all_scores, all_labels)
    mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, dcf_p_target, dcf_c_miss, dcf_c_fa)

    print('\n', time.strftime("%Y-%m-%d %H:%M:%S"), "VEER {:2.4f}".format(result[1]), "MinDCF {:2.5f}".format(mindcf))

    # 写入结果  
    scorefile.write("VEER {:2.4f}, MinDCF {:2.5f}\n".format(result[1], mindcf))
    scorefile.close()

    # print("\n")
    # i = 0
    # count = 0
    # avg_target_score = 0
    # avg_ori_score = 0
    # while i < len(all_scores):
    #     if all_scores[i] < all_scores[i+1]:
    #         count += 1
    #     avg_ori_score -= all_scores[i]
    #     avg_target_score -= all_scores[i+1]
    #     i += 2
    # print("avg target dist:", avg_target_score/(len(all_scores)/2))
    # print("avg ori dist:", avg_ori_score/(len(all_scores)/2))
    # print("convert rate:", count/(len(all_scores)/2))

    return (all_scores, all_labels, all_trials)


def evaluateFromList_si_compare(s, sap, test_list, nDataLoaderThread, eval_frames, result_save_path, epoch, enroll, hidden_size, batch_size, perb_uni, snr, target, result_file, adv_attack=True, model_attack=True, attack=None, **kwargs):

    top1 = 0
    top10 = 0
    dist_self = 0
    t_snr = 0
    t_pesq = 0
    p = 0


    #定义测试数据加载器
    test_dataset = test_dataset_loader(test_list, eval_frames=eval_frames, enroll=enroll, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = nDataLoaderThread,
        drop_last = True
    )

    #target_label = test_dataset.__target_label__(target)

    for data, data_label in test_loader:
        if attack == None:
            # 预测器生成扰动
            lookback = data[:, :enroll].unsqueeze(1).cuda()
            with torch.no_grad():
                _, perb = sap(perb_uni, lookback)
                perb = perb.squeeze(1)
        else:
            # 种子扰动
            perb = perb_uni.squeeze(1)
        

        if attack == "rir":
            perb_amp = torch_conv(data[:, int(0.5*enroll):int(1.5*enroll)].cuda(), perb).squeeze(1) - data[:, int(0.5*enroll):int(1.5*enroll)].cuda()
            origin_l2 = torch.sum(data[:, int(0.5*enroll):int(1.5*enroll)].cuda()**2)
            power_perb = origin_l2 / (10.0**(snr/10))
            perb_l2 = torch.sum(perb_amp**2)
            perb = perb_amp * torch.sqrt((power_perb / perb_l2))

        else:
            perb_amp = perb
            origin_l2 = torch.sum(data[:, int(0.5*enroll):int(1.5*enroll)].cuda()**2)
            power_perb = origin_l2 / (10.0**(snr/10))
            perb_l2 = torch.sum(perb_amp**2)
            perb = perb_amp * torch.sqrt((power_perb / perb_l2))

        adv = data[:, int(0.5*enroll):int(1.5*enroll)].cuda() + perb

        adv_feat = s(adv)
        speaker_feature = s(data[:, :enroll])  # 计算与自身特征距离时用
        dist = torch.pairwise_distance(speaker_feature, adv_feat)

        snr_outs = snr_out(data[:, int(0.5*enroll):int(1.5*enroll)].cuda(), perb)
        try:
            p = pesq(16000, data[:, int(0.5*enroll):int(1.5*enroll)][0].detach().cpu().numpy(), adv[0].detach().cpu().numpy())
        except:
            p = p
        t_pesq += p

        # # 随机重采样
        # start = random.randint(0, enroll)
        # adv = adv[:, start:start+enroll]

        # 计算top1, top10, snr, pesq
        #target_label_tensor = torch.tensor(data.size(0)*[target_label])
        _, prec = s(adv, data_label.cuda())
        top1 += torch.sum(prec[0]).detach().cpu().item()
        top10 += torch.sum(prec[1]).detach().cpu().item()
        dist_self += torch.mean(dist).detach().cpu().item()
        t_snr += snr_outs.detach().cpu().item()

    # 计算平均输出
    sys.stdout.write("Epoch: {:d}, Top1: {:2.3f}%, Top10: {:2.3f}%, Dist: {:2.3f}, SNR: {:2.3f}, PESQ: {:2.3f}".format(epoch, top1/len(test_loader), top10/len(test_loader), dist_self/len(test_loader), t_snr/len(test_loader), t_pesq/len(test_loader)))
    sys.stdout.flush()

    # 写入结果
    with open(result_file, 'a') as f:
        f.write("Epoch: {:d}, Top1: {:2.3f}%, Top10: {:2.3f}%, Dist: {:2.3f}, SNR: {:2.3f}, PESQ: {:2.3f}\n".format(epoch, top1/len(test_loader), top10/len(test_loader), dist_self/len(test_loader), t_snr/len(test_loader), t_pesq/len(test_loader)))

    print("\n")
    return 



def evaluateFromList_sv_compare(s, sap, test_list, nDataLoaderThread, eval_frames, elen, result_save_path, hidden_size, dcf_p_target, dcf_c_miss, dcf_c_fa, result_file=None, perb_uni=None, print_interval=100, num_eval=4, attack=None, **kwargs):

    lines = []
    files = []
    feats_ori = {}
    feats_adv = {}
    tstart = time.time()
    scorefile = open(result_save_path, "a")

    with open(test_list) as f:
        lines = f.readlines()
    
    #测试样本list
    files = list(itertools.chain(*[x.strip().split()[-2:]  for x in lines]))
    setfiles = list(set(files))
    setfiles.sort()

    #定义测试数据加载器
    test_dataset = test_sv_dataset_loader(setfiles, num_eval=num_eval, eval_frames=eval_frames, enroll=elen, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = nDataLoaderThread,
        drop_last = False
    )

    #给每个测试样本提取特征
    adv1 = torch.zeros(num_eval, elen).cuda()
    adv1 = adv1.reshape(adv1.size()[0], int(adv1.size()[1]/hidden_size), hidden_size)
    t_attn_mask = generate_square_subsequent_mask(adv1.size()[1])

    for idx, data in enumerate(test_loader):
        #print(data[0].shape)
        #print(data[1])
        inp = data[0].cuda()
        inp = inp.permute(1, 0, 2)
        with torch.no_grad():
            inp_ori = inp[:, :, :elen]
            ref_feat_ori = s(inp_ori.squeeze(1)).detach().cuda()

            inp1 = inp[:, :, :elen]
            inp2 = inp[:, :, elen:2*elen]
            # 生成perb
            if attack == None:
                # 预测器生成扰动
                lookback = inp1
                with torch.no_grad():
                    _, perb = sap(perb_uni, lookback)
                    perb = perb.squeeze(1)
            else:
                # 种子扰动
                perb = perb_uni.squeeze(1)
                perb = perb[0].repeat(inp2.size(0), 1)


            if attack == "rir":
                perb_amp = torch_conv(inp2.squeeze(1).cuda(), perb).squeeze(1) - inp2.squeeze(1).cuda()
                origin_l2 = torch.sum(inp2.squeeze(1).cuda()**2)
                power_perb = origin_l2 / (10.0**(15/10))
                perb_l2 = torch.sum(perb_amp**2)
                perb = perb_amp * torch.sqrt((power_perb / perb_l2))

            else:
                perb_amp = perb
                origin_l2 = torch.sum(inp2.squeeze(1).cuda()**2)
                power_perb = origin_l2 / (10.0**(20/10))
                perb_l2 = torch.sum(perb_amp**2)
                perb = perb_amp * torch.sqrt((power_perb / perb_l2))

            adv2 = inp2.squeeze(1) + perb
            ref_feat_adv = s(adv2).detach().cuda()

        feats_ori[data[1][0]] = ref_feat_ori
        feats_adv[data[1][0]] = ref_feat_adv
        telapsed = time.time() - tstart

        if idx % print_interval == 0:
            sys.stdout.write("\r Reading {:d} of {:d}: {:.2} Hz, embedding size {:d}".format(idx, test_loader.__len__(), idx/telapsed, ref_feat_ori.size()[1]))


    print("\n")
    all_scores = []
    all_labels = []
    all_trials = []

    tstart = time.time()

    for idx, line in enumerate(lines):
        data = line.split()
        
        feat1 = feats_adv[data[1]].cuda()
        feat2 = feats_ori[data[2]].cuda()

        #正则化特征
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)
        #dist = F.pairwise_distance(feat1.unsqueeze(-1), feat2.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy()
        dist = torch.pairwise_distance(feat1, feat2)
        dist = torch.mean(dist).detach().cpu().numpy()
        score = -1 * dist

        all_scores.append(score)
        all_labels.append(int(data[0]))
        all_trials.append(data[1] + " " + data[2])

        if idx % print_interval == 0:
            telapsed = time.time() - tstart
            sys.stdout.write("\r Computing {:d} of {:d}".format(idx, len(lines)))
            sys.stdout.flush()

    # 计算平均输出
    result = tuneThresholdfromScore(all_scores, all_labels, [1,0,1])
    fnrs, fprs, thresholds = ComputeErrorRates(all_scores, all_labels)
    mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, dcf_p_target, dcf_c_miss, dcf_c_fa)

    print('\n', time.strftime("%Y-%m-%d %H:%M:%S"), "VEER {:2.4f}".format(result[1]), "MinDCF {:2.5f}".format(mindcf))

    # # 写入结果  
    # scorefile.write("VEER {:2.4f}, MinDCF {:2.5f}\n".format(result[1], mindcf))
    # scorefile.close()
    # 写入结果
    with open(result_file, 'a') as f:
        f.write("VEER {:2.4f}, MinDCF {:2.5f}\n".format(result[1], mindcf))

    # print("\n")
    # i = 0
    # count = 0
    # avg_target_score = 0
    # avg_ori_score = 0
    # while i < len(all_scores):
    #     if all_scores[i] < all_scores[i+1]:
    #         count += 1
    #     avg_ori_score -= all_scores[i]
    #     avg_target_score -= all_scores[i+1]
    #     i += 2
    # print("avg target dist:", avg_target_score/(len(all_scores)/2))
    # print("avg ori dist:", avg_ori_score/(len(all_scores)/2))
    # print("convert rate:", count/(len(all_scores)/2))

    return (all_scores, all_labels, all_trials)

def evaluateFromList_si_compare(s, sap, test_list, nDataLoaderThread, eval_frames, result_save_path, epoch, enroll, hidden_size, batch_size, perb_uni, snr, target, result_file, adv_attack=True, model_attack=True, attack=None, **kwargs):

    top1 = 0
    top10 = 0
    dist_self = 0
    t_snr = 0
    t_pesq = 0
    p = 0


    #定义测试数据加载器
    test_dataset = test_dataset_loader(test_list, eval_frames=eval_frames, enroll=enroll, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = nDataLoaderThread,
        drop_last = True
    )

    #target_label = test_dataset.__target_label__(target)

    for data, data_label in test_loader:
        if attack == None:
            # 预测器生成扰动
            lookback = data[:, :enroll].unsqueeze(1).cuda()
            with torch.no_grad():
                _, perb = sap(perb_uni, lookback)
                perb = perb.squeeze(1)
        else:
            # 种子扰动
            perb = perb_uni.squeeze(1)
        

        if attack == "rir":
            perb_amp = torch_conv(data[:, int(0.5*enroll):int(1.5*enroll)].cuda(), perb).squeeze(1) - data[:, int(0.5*enroll):int(1.5*enroll)].cuda()
            origin_l2 = torch.sum(data[:, int(0.5*enroll):int(1.5*enroll)].cuda()**2)
            power_perb = origin_l2 / (10.0**(snr/10))
            perb_l2 = torch.sum(perb_amp**2)
            perb = perb_amp * torch.sqrt((power_perb / perb_l2))

        else:
            perb_amp = perb
            origin_l2 = torch.sum(data[:, int(0.5*enroll):int(1.5*enroll)].cuda()**2)
            power_perb = origin_l2 / (10.0**(snr/10))
            perb_l2 = torch.sum(perb_amp**2)
            perb = perb_amp * torch.sqrt((power_perb / perb_l2))

        adv = data[:, int(0.5*enroll):int(1.5*enroll)].cuda() + perb

        adv_feat = s(adv)
        speaker_feature = s(data[:, :enroll])  # 计算与自身特征距离时用
        dist = torch.pairwise_distance(speaker_feature, adv_feat)

        snr_outs = snr_out(data[:, int(0.5*enroll):int(1.5*enroll)].cuda(), perb)
        try:
            p = pesq(16000, data[:, int(0.5*enroll):int(1.5*enroll)][0].detach().cpu().numpy(), adv[0].detach().cpu().numpy())
        except:
            p = p
        t_pesq += p

        # # 随机重采样
        # start = random.randint(0, enroll)
        # adv = adv[:, start:start+enroll]

        # 计算top1, top10, snr, pesq
        #target_label_tensor = torch.tensor(data.size(0)*[target_label])
        _, prec = s(adv, data_label.cuda())
        top1 += torch.sum(prec[0]).detach().cpu().item()
        top10 += torch.sum(prec[1]).detach().cpu().item()
        dist_self += torch.mean(dist).detach().cpu().item()
        t_snr += snr_outs.detach().cpu().item()

    # 计算平均输出
    sys.stdout.write("Epoch: {:d}, Top1: {:2.3f}%, Top10: {:2.3f}%, Dist: {:2.3f}, SNR: {:2.3f}, PESQ: {:2.3f}".format(epoch, top1/len(test_loader), top10/len(test_loader), dist_self/len(test_loader), t_snr/len(test_loader), t_pesq/len(test_loader)))
    sys.stdout.flush()

    # 写入结果
    with open(result_file, 'a') as f:
        f.write("Epoch: {:d}, Top1: {:2.3f}%, Top10: {:2.3f}%, Dist: {:2.3f}, SNR: {:2.3f}, PESQ: {:2.3f}\n".format(epoch, top1/len(test_loader), top10/len(test_loader), dist_self/len(test_loader), t_snr/len(test_loader), t_pesq/len(test_loader)))

    print("\n")
    return 
    


def loadWAV_long(filenames, audio_len=972800):
    # filenames包含5条样本
    # 读取音频文件
    audios = np.array([])
    for file in filenames:
        audio, sample_rate = sf.read(file)
        audios = np.append(audios, audio)
    
    #如果音频长度小于最大长度则补齐
    audiosize = audios.shape[0]
    if audiosize <= audio_len:
        shortage = audio_len - audiosize + 1
        audios = np.pad(audios, (0, shortage), 'constant')
        audiosize = audios.shape[0]
    else:
        audios = audios[:audio_len]

    return audios


class long_loader(Dataset):

    def __init__(self, test_list, audio_len, **kwargs):
        self.audio_len = audio_len
        #读取训练集列表
        with open(test_list) as dataset_file:
            lines = dataset_file.readlines()

        #构建一个字典,key是说话人的名字,value是说话人的序号
        self.dictkeys = list(set([x.split()[0] for x in lines]))
        self.dictkeys.sort()
        self.dictkeys = {key : ii for ii, key in enumerate(self.dictkeys)}

        #构建一个数据集路径的列表,和一个数据集标签的列表
        self.data_list = []
        self.data_label = []

        for lidx, line in enumerate(lines):
            data = line.strip().split()
            filenames = [data[1], data[2], data[3], data[4], data[5]]
            #说话人标签
            speaker_label = self.dictkeys[data[0]]

            self.data_label.append(speaker_label)
            self.data_list.append(filenames)

    def __getitem__(self, indice):

        audio = loadWAV_long(self.data_list[indice], self.audio_len)
        return torch.FloatTensor(audio).cuda(), self.data_label[indice]

    def __len__(self):
        #可以len(class)查看数据量
        return len(self.data_list)
    


def evaluateFromList_si_long(s, sap, test_list, nDataLoaderThread, eval_frames, result_save_path, epoch, enroll, hidden_size, batch_size, perb_uni, snr, target, result_file, num, adv_attack=True, model_attack=True, attack=None, **kwargs):
    snrs = [45, 40, 35, 30, 25]
    perb_uni = perb_uni[0]
    for snr in snrs:
        top1 = 0
        top10 = 0
        audio_len = 972800
        
        test_list = "I:\\paper2\\MAJOR\\CODES\\datas\\Libri_test_long_list.txt"


        #定义测试数据加载器
        test_dataset = long_loader(test_list, audio_len, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size = 1,
            shuffle = False,
            num_workers = nDataLoaderThread,
            drop_last = True
        )

        count = 0
        p = 0
        p_num = 0
        num_dis = np.arange(0, 25)
        np.random.shuffle(num_dis)
        for data, data_label in test_loader:
            count += 1
            # 生成对抗样本并将原样本和对抗样本都保存
            adv = []
            #adv.append(data[:, :enroll] + perb_uni.squeeze(1))
            for ii in range(int(audio_len/enroll)):
                
                if attack == None or attack == "vbm":
                    # 预测器生成扰动
                    lookback = data[:, (ii)*enroll:(ii+1)*enroll].unsqueeze(1).cuda()
                    with torch.no_grad():
                        _, perb = sap(perb_uni.unsqueeze(0), lookback)
                        perb = perb.squeeze(1)
                else:
                    # 种子扰动
                    perb = perb_uni.squeeze(1)

                if attack == "rir":
                    perb_amp = torch_conv(data[:, ii*enroll:(ii+1)*enroll].cuda(), perb).squeeze(1) - data[:, ii*enroll:(ii+1)*enroll].cuda()
                    origin_l2 = torch.sum(data[:, ii*enroll:(ii+1)*enroll].cuda()**2)
                    power_perb = origin_l2 / (10.0**(snr/10))
                    perb_l2 = torch.sum(perb_amp**2)
                    perb = perb_amp * torch.sqrt((power_perb / perb_l2))

                else:
                    perb_amp = perb
                    origin_l2 = torch.sum(data[:, ii*enroll:(ii+1)*enroll].cuda()**2)
                    power_perb = origin_l2 / (10.0**(snr/10))
                    perb_l2 = torch.sum(perb_amp**2)
                    perb = perb_amp * torch.sqrt((power_perb / perb_l2))

                adv.append(data[:, ii*enroll:(ii+1)*enroll].cuda() + perb)
                # 计算PESQ
                try:
                    p += pesq(16000, data[0, ii*enroll:(ii+1)*enroll].detach().cpu().numpy(), (data[0, ii*enroll:(ii+1)*enroll].cuda() + perb[0]).detach().cpu().numpy())
                    p_num += 1
                except:
                    p += 0


            adv = torch.stack(adv, dim=1)
            adv = adv.reshape(adv.size(0), -1)

            # 随机采样10条样本进行测试
            adv_test = []
            for jj in range(10):
                start = random.randint(0, audio_len-enroll)
                adv_test.append(adv[:, start:start+enroll])
            adv_test = torch.stack(adv_test, dim=1)

            data_label = data_label.repeat(10)
            _, prec = s(adv_test.squeeze(0), data_label.cuda())
            top1 += torch.sum(prec[0]).detach().cpu().item()
            top10 += torch.sum(prec[1]).detach().cpu().item()

            if attack == None:
                attack = "ours"

            # 保存原始样本和对抗样本
            num += 1
            path = ".\\result_major\\subject_label\\" + str(snr) + "_" + str(attack)
            if not os.path.exists(path):
                os.makedirs(path)
            path_ori = "\\" + str(int(num)) + "_" + "ori" + "_" + str(int(data_label[0].detach().cpu().item())) + ".wav"
            path_adv = "\\" + str(int(num)) + "_" + "adv" + "_" + str(int(data_label[0].detach().cpu().item())) + "_" + attack + "_" + str(int(snr)) + ".wav"
            sf.write(path+path_ori, data[0].detach().cpu().numpy(), samplerate=16000)
            sf.write(path+path_adv, adv[0].detach().cpu().numpy(), samplerate=16000)
            # # 从每条对抗样本和原始样本中各采集两个长度为80000的样本，保存在答案文件夹
            # a = num_dis[2*count]
            # b = num_dis[2*count-1]
            # for jj in range(2):
            #     path_ori = "\\" + str(int(num)) + "_" + str(a) + "_" + "ori" + "_" + str(jj) + "_" + str(int(data_label[0].detach().cpu().item())) + ".wav"
            #     path_adv = "\\" + str(int(num)) + "_" + str(b) + "_" + "adv" + "_" + str(jj) + "_" + str(int(data_label[0].detach().cpu().item())) + "_" + attack + "_" + str(int(snr)) + ".wav"
            #     start = random.randint(0, audio_len-80000)
            #     sf.write(path+path_adv, adv[:, start:start+80000][0].detach().cpu().numpy(), samplerate=16000)
            #     start = random.randint(0, audio_len-80000)
            #     sf.write(path+path_ori, data[:, start:start+80000][0].detach().cpu().numpy(), samplerate=16000)

            # path_test = ".\\result_major\\subject_test\\" + str(snr) + "_" + str(attack)
            # if not os.path.exists(path_test):
            #     os.makedirs(path_test)
            # path_adv = "\\" + str(int(num)) + ".wav"
            # sf.write(path_test+path_adv, adv[0].detach().cpu().numpy(), samplerate=16000)

            # # 从每条对抗样本和原始样本中各采集两个长度为80000的样本，保存在测试文件夹
            # for jj in range(2):
            #     path_ori = "\\" + str(a) + "_" + str(jj) + ".wav"
            #     path_adv = "\\" + str(b) + "_" + str(jj) + ".wav"
            #     start = random.randint(0, audio_len-80000)
            #     sf.write(path_test+path_adv, adv[:, start:start+80000][0].detach().cpu().numpy(), samplerate=16000)
            #     start = random.randint(0, audio_len-80000)
            #     sf.write(path_test+path_ori, data[:, start:start+80000][0].detach().cpu().numpy(), samplerate=16000)

            if count > 1:
                break

        # 计算平均输出
        sys.stdout.write("Epoch: {:d}, Top1: {:2.3f}%, Top10: {:2.3f}%, pesq:{:2.4f}".format(epoch, top1/12, top10/12, p/p_num))
        sys.stdout.flush()

        # 写入结果
        with open(result_file, 'a') as f:
            f.write("Epoch: {:d}, Top1: {:2.3f}%, Top10: {:2.3f}%, pesq:{:2.4f}\n".format(epoch, top1/12, top10/12, p/p_num))

        print("\n")
    return num

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

# def pesq_avg(signal_source, signal_noise):
#     for 
