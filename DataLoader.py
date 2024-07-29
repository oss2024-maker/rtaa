# -*- encoding: utf-8 -*-
#数据集的处理与加载

import glob
import os
import random
from csv import list_dialects
from turtle import st

import numpy as np
import soundfile
import torch
import torch.distributed as dist
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import DataLoader, Dataset


def round_down(num, divisor):
    return num - (num%divisor)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

#单个音频加载函数
def loadWAV(filename, max_frames, evalmode=True, num_eval=10):

    #定义最大音频长度
    max_audio = max_frames * 160 + 240

    #读取音频文件并将其转换成tensor
    audio, sample_rate = soundfile.read(filename)
    audiosize = audio.shape[0]

    #如果音频长度小于最大长度则补齐
    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), 'constant')
        audiosize = audio.shape[0]
    
    if evalmode:
        #测试时采用等距方法选取开始位置
        startframe = np.int64(np.linspace(0, audiosize-max_audio, num=num_eval))
    else:
        #随机选取开始的位置  
        startframe = np.array([np.int64(random.random()*(audiosize-max_audio))])

    feats = []
    for asf in startframe:
        feats.append(audio[int(asf):int(asf)+max_audio])

    #把选中的音频段堆起来
    feat = np.stack(feats, axis=0)

    return feat


#数据集增强类
class AugmentWAV(object):

    def __init__(self, musan_path, rir_path, max_frames):

        self.max_frames = max_frames
        self.max_audio  = max_frames * 160 + 240

        self.noisetypes = ['noise','speech','music']

        self.noisesnr   = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise   = {'noise':[1,1], 'speech':[3,7],  'music':[1,1] }
        self.noiselist  = {}

        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'));

        for file in augment_files:
            if not file.split('\\')[-4] in self.noiselist:
                self.noiselist[file.split('\\')[-4]] = []
            self.noiselist[file.split('\\')[-4]].append(file)

        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'));

    def additive_noise(self, noisecat, audio):

        clean_db = 10 * np.log10(np.mean(audio ** 2)+1e-4) 

        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        noises = []

        for noise in noiselist:

            noiseaudio  = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2)+1e-4) 
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)

        return np.sum(np.concatenate(noises,axis=0),axis=0,keepdims=True) + audio

    def reverberate(self, audio):

        rir_file    = random.choice(self.rir_files)
        
        rir, fs     = soundfile.read(rir_file)
        rir         = np.expand_dims(rir.astype(np.float),0)
        rir         = rir / np.sqrt(np.sum(rir**2))
        #print('audio:', audio.shape)
        #print('rir:', rir.shape)
        return signal.convolve(audio, rir, mode='full')[:,:self.max_audio]


#训练数据集类,可迭代调用,无采样器,和数据增强
class train_dataset_loader_wosampler(Dataset):
    
    def __init__(self, train_list, max_frames, **kwargs):

        self.train_list = train_list
        self.max_frames = max_frames

        #读取训练集列表
        with open(train_list) as dataset_file:
            lines = dataset_file.readlines()

        #构建一个字典,key是说话人的名字,value是说话人的序号
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = {key : ii for ii, key in enumerate(dictkeys)}

        #构建一个数据集路径的列表,和一个数据集标签的列表
        self.data_list = []
        self.data_label = []

        for lidx, line in enumerate(lines):
            data = line.strip().split()

            #说话人标签
            speaker_label = dictkeys[data[0]]
            #语音路径
            filename = data[1]

            self.data_label.append(speaker_label)
            self.data_list.append(filename)
    
    # def __getitem__(self, indices):
    #     #可以迭代调用,加载一个batch音频并拼接

    #     feat = []
        
    #     for index in indices:

    #         audio = loadWAV(self.data_list[index], self.max_frames)
    #         feat.append(audio)

    #     feat = np.concatenate(feat, axis=0)

    #     return torch.FloatTensor(feat), self.data_label[index]

    def __getitem__(self, indice):

        audio = loadWAV(self.data_list[indice], self.max_frames, evalmode=False)
        return torch.FloatTensor(audio), self.data_label[indice]

    def __len__(self):
        #可以len(class)查看数据量
        return len(self.data_list)

#训练数据集类,含数据集采样器
class train_dataset_loader(Dataset):
    def __init__(self, train_list, augment, musan_path, rir_path, max_frames, **kwargs):

        self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames = max_frames)

        self.train_list = train_list
        self.max_frames = max_frames;
        self.musan_path = musan_path
        self.rir_path   = rir_path
        self.augment    = augment
        
        # Read training files
        with open(train_list) as dataset_file:
            lines = dataset_file.readlines();

        # Make a dictionary of ID names and ID indices
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }

        # Parse the training list into file names and ID indices
        self.data_list  = []
        self.data_label = []
        
        for lidx, line in enumerate(lines):
            data = line.strip().split();

            speaker_label = dictkeys[data[0]];
            filename = data[1]
            
            self.data_label.append(speaker_label)
            self.data_list.append(filename)

    def __getitem__(self, indices):

        feat = []
        
        for index in indices:
            
            audio = loadWAV(self.data_list[index], self.max_frames, evalmode=False)
            
            if self.augment:
                augtype = random.randint(0,4)

                if augtype == 1:
                    audio   = self.augment_wav.reverberate(audio)
                elif augtype == 2:
                    audio   = self.augment_wav.additive_noise('music',audio)
                elif augtype == 3:
                    audio   = self.augment_wav.additive_noise('speech',audio)
                elif augtype == 4:
                    audio   = self.augment_wav.additive_noise('noise',audio)
                    
            feat.append(audio);

        feat = np.concatenate(feat, axis=0)
        #print(feat.shape)

        return torch.FloatTensor(feat), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


#测试数据集类
class test_dataset_loader(Dataset):

    def __init__(self, test_list, eval_frames, num_eval, **kwargs):
        self.max_frames = eval_frames
        self.num_eval = num_eval
        self.test_list = test_list

    def __getitem__(self, index):
        audio = loadWAV(self.test_list[index], self.max_frames, evalmode=True, num_eval=self.num_eval)

        return torch.FloatTensor(audio), self.test_list[index]

    def __len__(self):
        
        return len(self.test_list)

#测试数据集类
class test_si_dataset_loader(Dataset):
    
    def __init__(self, test_si_list, max_frames, **kwargs):

        self.train_list = test_si_list
        self.max_frames = max_frames

        #读取训练集列表
        with open(test_si_list) as dataset_file:
            lines = dataset_file.readlines()

        #构建一个字典,key是说话人的名字,value是说话人的序号
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = {key : ii for ii, key in enumerate(dictkeys)}

        #构建一个数据集路径的列表,和一个数据集标签的列表
        self.data_list = []
        self.data_label = []

        for lidx, line in enumerate(lines):
            data = line.strip().split()

            #说话人标签
            speaker_label = dictkeys[data[0]]
            #语音路径
            filename = data[1]

            self.data_label.append(speaker_label)
            self.data_list.append(filename)


    def __getitem__(self, indice):

        audio = loadWAV(self.data_list[indice], self.max_frames, evalmode=False)
        return torch.FloatTensor(audio), self.data_label[indice]

    def __len__(self):
        #可以len(class)查看数据量
        return len(self.data_list)
    
#数据集采样方法

class train_dataset_sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size, seed, **kwargs):

        self.data_label         = data_source.data_label;
        self.nPerSpeaker        = nPerSpeaker;
        self.max_seg_per_spk    = max_seg_per_spk;
        self.batch_size         = batch_size;
        self.epoch              = 0;
        self.seed               = seed;
        
    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        #随机打乱
        indices = torch.randperm(len(self.data_label), generator=g).tolist()

        data_dict = {}

        #构建一个字典,用每个说话人的序号作为key,value是一个list,包含这个说话人的utter的index
        for index in indices:
            speaker_label = self.data_label[index]
            if not (speaker_label in data_dict):
                data_dict[speaker_label] = [];
            data_dict[speaker_label].append(index);


        #把说话人构成一个list
        dictkeys = list(data_dict.keys());
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []
        
        for findex, key in enumerate(dictkeys):
            data    = data_dict[key]
            #round_down是让每个说话人的总句子数可以整除nPerSpeaker
            numSeg  = round_down(min(len(data),self.max_seg_per_spk),self.nPerSpeaker)
            #把numSeg按照nPerSpeaker分割
            rp      = lol(np.arange(numSeg),self.nPerSpeaker)
            #flattened_label是每个nPerSpeaker对应说话人的序号
            flattened_label.extend([findex] * (len(rp)))
            #flattened_list
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        ## Mix data in random order
        mixid           = torch.randperm(len(flattened_label), generator=g).tolist()
        mixlabel        = []
        mixmap          = []

        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = round_down(len(mixlabel), self.batch_size)
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        mixed_list = [flattened_list[i] for i in mixmap]


        total_size = round_down(len(mixed_list), self.batch_size)
        self.num_samples = total_size
        return iter(mixed_list[:total_size])

    
    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch








