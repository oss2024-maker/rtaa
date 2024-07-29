# 主观测试
import glob
import os
import random
import warnings

import numpy as np
import soundfile as sf
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from wave_unet_attention2 import Attention_Waveunet
from UNET_util4 import evaluateFromList_si_compare, evaluateFromList_sv_compare, evaluateFromList_si_long, loadWAV
from Speaker_net import *
from spr_utils import *
from compare_attack import seed_gen_raa
from compare_attack2 import *
from pesq import pesq
from Voicebox import VoiceBox


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

class TIMIT(Dataset):
    def __init__(self, files, enroll, data_label, max_frames=200):

        self.max_frames = max_frames
        self.enroll = enroll
        self.files = files
        self.data_label = data_label

    def __getitem__(self, index):
        origin = loadWAV(self.files[index][1], max_frames=self.max_frames, enroll=self.enroll)
        origin_audio = torch.from_numpy(origin).float()

        return origin_audio, self.data_label[index]
    
    def __len__(self):
        return len(self.files)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def main(hp, num):

    # 初始化SAP模型
    num_features = [hp.features*i for i in range(1, hp.levels+1)]
    if hp.compare == "vbm":
        sap = VoiceBox().cuda()
    else:
        sap = Attention_Waveunet(snr=hp.snr, num_inputs=hp.channels, num_channels=num_features, num_outputs=hp.channels, enroll=hp.enroll).cuda()
    if hp.load_sap == True:
        self_state = sap.state_dict()
        loaded_state = torch.load(hp.sapmodel_path, map_location="cuda:0")
        for name, param in loaded_state.items():
            self_state[name].copy_(param)
        if self_state.keys() == loaded_state.keys():
            print("sapmodel has loaded!")

    # 初始化说话人识别模型
    s = SpeakerNet(**hp).cuda()
    self_state = s.state_dict()
    loaded_state = torch.load(hp.svmodel_path, map_location="cuda:0")
    for name, param in loaded_state.items():
        self_state[name].copy_(param)
    if self_state.keys() == loaded_state.keys():
        print("svmodel has loaded!")
    s.eval()

    # 初始化目标模型
    t = SpeakerNet(model=hp.tmodel, trainfunc=hp.trainfunc, n_out=hp.n_out, n_class=hp.n_class, nPerSpeaker=hp.nPerSpeaker).cuda()
    self_state = t.state_dict()
    loaded_state = torch.load(hp.tmodel_path, map_location="cuda:0")
    for name, param in loaded_state.items():
        self_state[name].copy_(param)
    if self_state.keys() == loaded_state.keys():
        print("tmodel has loaded!")
    t.eval()

    # 初始化数据加载器
    with open(hp.data_list) as dataset_file:
        lines = dataset_file.readlines()
    # files是文件名对(ori, adv)组成的列表
    files = []
    for line in lines:
        file = line.strip().split(' ')
        files.append(file)

    #读取训练集列表
    with open(hp.data_list) as dataset_file:
        lines = dataset_file.readlines()

    #构建一个字典,key是说话人的名字,value是说话人的序号
    dictkeys = list(set([x.split()[0] for x in lines]))
    dictkeys.sort()
    dictkeys = {key : ii for ii, key in enumerate(dictkeys)}

    #构建一个数据集路径的列表,和一个数据集标签的列表
    data_list = []
    data_label = []

    for lidx, line in enumerate(lines):
        data = line.strip().split()

        #说话人标签
        speaker_label = dictkeys[data[0]]
        #语音路径
        filename = data[1]

        data_label.append(speaker_label)
        data_list.append(filename)

    train_dataset = TIMIT(files, enroll=hp.enroll, max_frames=hp.eval_frames, data_label=data_label)
    train_loader  = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = hp.batch_size,
        num_workers = hp.nDataLoaderThead,
        pin_memory = False,
        shuffle = False,
        worker_init_fn = worker_init_fn,
        drop_last = True,
    )
    train_loader_seed  = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = 128,
        num_workers = hp.nDataLoaderThead,
        pin_memory = False,
        shuffle = True,
        worker_init_fn = worker_init_fn,
        drop_last = True,
    )

    # 定义优化器
    optimizer = Optimizer_SGD(sap.parameters(), **hp)

    # 定义学习率衰减
    scheduler,  _ = Scheduler(optimizer, **hp)

    # 结果保存路径
    result_save_path = ".\\results\\compare\\" + hp.model + "_" + "unet2" + ".txt"

    # 设定目标
    target = "TEST_DR6_FMGD0"
    path=".\\adv_samples_" + target
    oringin = "I:\\datas\\TIMIT\\" + target.split("_")[0] + "\\" + target.split("_")[1] + "\\" + target.split("_")[2]

    # 加载目标说话人的数据
    target_files = glob.glob(oringin + "\\" + "*.WAV")
    feats = []
    for file in target_files:
        # 对目标说话人提取特征
        audio = loadWAV(file, max_frames=400, enroll=25600)
        audio = torch.from_numpy(audio).float().cuda()
        feats.append(audio)
    feats = torch.stack(feats, axis=0)
    feats = s(feats)
    #feats = F.normalize(feats, p=2, dim=1)

    # 测试目标说话人特征距离
    dist = 0
    for i in range(feats.size()[0]):
        # arr1 = feats[0:i]
        # arr2 = feats[i+1:]
        # feat_others = torch.cat((arr1,arr2),dim=0)
        f1 = feats[1] 
        f2 = feats[i]
        dist += F.pairwise_distance(f1, f2)
    dist = dist / (feats.size()[0])
    print("目标样本内特征平均距离:", dist)
    
    # 如果使用对比的攻击
    if hp.compare != None and hp.compare != "vbm":
        if hp.compare == "rir":
            method = RIR(result_file=hp.result_path)
        elif hp.compare == "advpulse":
            method = ADVPULSE(result_file=hp.result_path)
        elif hp.compare == "eon":
            method = EON(result_file=hp.result_path)
        elif hp.compare == "pgd":
            method = PGD(result_file=hp.result_path)
        elif hp.compare == "uni":
            method = UNI(result_file=hp.result_path)
        elif hp.compare == "gn":
            method = GN(result_file=hp.result_path)

        if hp.compare == "rir":
            # 仅rirattack使用
            rir_files  = glob.glob(os.path.join(hp.rir_files,'*\\*\\*.wav'))
            rir_file = random.choices(rir_files, k=1)[0]
            rir, fs = sf.read(rir_file)
            rir = np.expand_dims(rir.astype(np.float32), 0)
            rir = np.tile(rir, (1, 20))
            asf = np.array([np.int64(random.random()*(rir.shape[1]-hp.enroll))])
            rir = rir[:, int(asf):int(asf)+hp.enroll]
            rir = rir / np.sqrt(np.sum(rir**2))
            perb = torch.tensor(rir[0]).cuda()
        else:
            perb = 0.0001 * torch.randn(hp.enroll).cuda()

        count = 1
        for data, data_label in train_loader_seed:
            count += 1
            origin_data = data[:, hp.enroll:2*hp.enroll].cuda()
            perb = method.attack(origin_data, s, 100, perb, feats, lr=0.0005, label=data_label)
            if hp.compare == "pgd":
                break
            if count > 10:
                break

        # 进行测试
        perb = perb.repeat(hp.batch_size, 1, 1)
        num = evaluateFromList_si_long(s, sap, hp.test_list, hp.nDataLoaderThead, hp.eval_frames, num=num, result_save_path=result_save_path, epoch=0, result_file=hp.result_path, snr=hp.snr, target="FMGD0", enroll=hp.enroll, hidden_size=hp.hidden_size, batch_size=hp.batch_size, perb_uni=perb, model_attack=False, attack=hp.compare)      
        #evaluateFromList_sv_compare(s, sap, hp.test_sv_list, hp.nDataLoaderThead, hp.eval_frames, elen=hp.enroll, attack=hp.compare, result_save_path=result_save_path, result_file=hp.result_path, perb_uni=perb, hidden_size=hp.hidden_size, dcf_p_target=hp.dcf_p_target, dcf_c_miss=hp.dcf_c_miss, dcf_c_fa=hp.dcf_c_fa)

        return
    
    # 生成uni_perb
    if hp.zero_init == True:
        perb_uni = 0.0001*torch.randn(hp.enroll).cuda()

    else:
        perb_uni = 0.0001*torch.randn(hp.enroll)
        perb_uni = perb_uni.cuda()

        for i in range(1):
            print("epoch:", i)
            count = 1
            #perb_uni_all = perb_uni
            for data, data_label in train_loader_seed:
                count += 1

                origin_data = data[:, hp.enroll:2*hp.enroll].cuda()
                perb_uni = seed_gen_raa(origin_data, s, 100, perb_uni, feats, lr=0.0005)
                #perb_uni = rir_attack(origin_data, s, 100, perb_uni, feats, lr=0.0001)
                #perb_uni = advpulse_attack(origin_data, s, 100, perb_uni, feats, lr=0.0005)
                if count > 20:
                    break
            # #perb_uni = perb_uni / count

    
    perb_uni = perb_uni.repeat(hp.batch_size, 1, 1)
    #evaluateFromList_si(s, sap, hp.test_list, hp.nDataLoaderThead, hp.eval_frames, result_save_path=result_save_path, epoch=0, snr=hp.snr, target="FMGD0", enroll=hp.enroll, hidden_size=hp.hidden_size, batch_size=hp.batch_size, perb_uni=perb_uni, model_attack=False)      
    #evaluateFromList_si(t, sap, hp.test_list, hp.nDataLoaderThead, hp.eval_frames, result_save_path=result_save_path, epoch=0, snr=hp.snr, target="FMGD0", enroll=hp.enroll, hidden_size=hp.hidden_size, batch_size=hp.batch_size, perb_uni=perb_uni, model_attack=False)
    # 开始训练
    it = 1

    for it in range(it, hp.max_epoch):
        sap.train()
        aloss = 0
        ploss = 0
        nloss = 0
        count = 0
        index = 0
        snr = 0 
        p = 0
        pesqt = 0
        total_len = train_loader.__len__() * hp.batch_size

        if hp.train == True:
            s.eval()
            sap.train()

            for data, data_label in train_loader:
                count += 1
                data = data.to('cuda', non_blocking=True)
                # 模型梯度清零
                sap.zero_grad()
                s.zero_grad()

                # 生成对抗扰动
                lookback = data[:, : hp.enroll].unsqueeze(1)  #前视信号设定为12800
                if hp.zero_init == False or hp.compare == "vbm":
                    _, perb = sap(perb_uni, lookback)
                else:
                    perb, _ = sap(perb_uni, lookback)   #计算预测和扰动 shape是[4, 12800]
                # lookback = data[:, hp.enroll:2*hp.enroll].unsqueeze(1)  #前视信号设定为12800
                # perb2 = sap(perb_uni, lookback).squeeze(1)   #计算预测和扰动 shape是[4, 12800]
                # perb = torch.cat([perb1, perb2], dim=1)

                perb = perb.squeeze(1)
                # 扰动约束
                origin_l2 = torch.sum(data[:, int(0.5*hp.enroll):int(1.5*hp.enroll)]**2)
                power_perb = origin_l2 / (10.0**(25/10)) # 这里控制噪音量
                perb_l2 = torch.sum(perb**2)
                perb = perb * torch.sqrt((power_perb / perb_l2))

                # # 扰动约束
                # origin_l2 = torch.sum(data[:, :2*hp.enroll]**2)
                # power_perb = origin_l2 / (10.0**(35/10)) # 这里控制噪音量
                # perb_uni_repeat = torch.cat([perb_uni.squeeze(1),perb_uni.squeeze(1)],dim=1)
                # perb_l2 = torch.sum(perb_uni_repeat**2)
                # perb_uni_repeat = perb_uni_repeat * torch.sqrt((power_perb / perb_l2))

                
                adv_sample = data[:, int(0.5*hp.enroll):int(1.5*hp.enroll)] + perb

                if hp.loss == "con":
                    # 计算特征差异损失
                    adv_feat = s(adv_sample).unsqueeze(0)
                    speaker_feature = s(data[:, int(0.5*hp.enroll):int(1.5*hp.enroll)])
                    mean_feature = torch.mean(speaker_feature, dim=0)
                    speaker_feature = mean_feature.repeat(hp.batch_size, 1)
                    dist = torch.pairwise_distance(adv_feat, speaker_feature.detach())
                    adv_loss = -torch.mean(dist)
                    n_loss = hp.alpha * adv_loss

                else:
                    #计算分类损失
                    adv_loss, prec = s(adv_sample, data_label.cuda())
                    adv_loss = -adv_loss
                    n_loss = 100*adv_loss

                # 反向传播并优化
                n_loss.backward()
                optimizer.step()

                # 打印损失
                _, prec = t(adv_sample, data_label.cuda())
                aloss += adv_loss.detach().cpu().item()
                ploss += torch.sum(prec[1]).detach().cpu().item()
                #nloss += n_loss.detach().cpu().item()
                index += 1

                #snr_outs = snr_out(data[:, hp.enroll:2*hp.enroll], perb)
                try:
                    p = pesq(16000, data[:, int(0.5*hp.enroll):int(1.5*hp.enroll)][0].detach().cpu().numpy(), adv_sample[0][:hp.enroll].detach().cpu().numpy())
                except:
                    p = p
                
                pesqt += p
                # if snr_outs > -50:
                #     snr += snr_outs

                sys.stdout.write("\rEpoch {:d}, processing {:d} of {:d}:".format(it, index*hp.batch_size, total_len))
                sys.stdout.write("aloss {:f}, pesq {:f}, prec {:4f}".format(aloss/index, pesqt/index, ploss/index))
                sys.stdout.flush()

                # with open(hp.result_path, 'a') as f:
                #     f.write("itr {:d}, pesq {:f}, prec {:f}\n".format(count, pesqt/index, ploss/index))


                if count >= 500:
                    break
            print("\n")
            scheduler.step()

            # 保存模型
            if not os.path.exists(hp.model_path):
                os.makedirs(hp.model_path)
            
            torch.save(sap.state_dict(), hp.model_path + "/sap%09d.model" % it)

        if it % hp.test_interval == 0:
            # 测试对抗样本与原始样本和目标样的特征的距离
            sap.eval()
            s.eval()
            t.eval()
            num = evaluateFromList_si_long(s, sap, hp.test_list, hp.nDataLoaderThead, hp.eval_frames, num=num, result_save_path=result_save_path, result_file=hp.result_path, attack=attack, epoch=it, snr=hp.snr, target="FMGD0", enroll=hp.enroll, hidden_size=hp.hidden_size, batch_size=hp.batch_size, perb_uni=perb_uni)      
            #evaluateFromList_si(t, sap, hp.test_list, hp.nDataLoaderThead, hp.eval_frames, result_save_path=result_save_path, epoch=it, snr=hp.snr, target="FMGD0", enroll=hp.enroll, hidden_size=hp.hidden_size, batch_size=hp.batch_size, perb_uni=perb_uni)
            #evaluateFromList_sv_compare(s, sap, hp.test_sv_list, hp.nDataLoaderThead, hp.eval_frames, elen=hp.enroll, attack=None, result_save_path=result_save_path, result_file=hp.result_path, perb_uni=perb_uni, hidden_size=hp.hidden_size, dcf_p_target=hp.dcf_p_target, dcf_c_miss=hp.dcf_c_miss, dcf_c_fa=hp.dcf_c_fa)
            if hp.train == False:
                break
    return num


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    # hp = Hparam(file='configs/config_Vox_targeted.yaml')
    # main(hp)

    # 主观测试
    # Libri-x-[rir,eon,advpulse,ours,vbm,gn,pgd,uni]
    hp = Hparam(file='configs/config_Libri_targeted.yaml')
    attacks = ["gn", "vbm", None, "rir", "eon", "advpulse"]
    # 开始测试
    hp.enroll = 25600
    hp.loss = "con"
    hp.svmodel_path = "I:\\paper2\\MAJOR\\CODES\\exps\\exp_LibriSpeech_X_vector_AAMSoftmax\\model\\model000000005.model" 
    hp.model = "X_vector"
    hp.tmodel_path = hp.svmodel_path
    hp.tmodel = hp.model
    path = ".\\result_major\\subject_test"
    hp.result_path = ".\\result_major\\subject_test\\Libri_X.txt"


    for attack in attacks:
        num = 0
        hp.compare = attack
        if not os.path.exists(path):
            os.makedirs(path)
        if attack == None:
            hp.zero_init = False
        else:
            hp.zero_init = True
        with open(hp.result_path, 'a') as f:
            f.write("dataset: Libri, loss:{}, attack:{}, model:{}, snr:{}\n".format(hp.loss, hp.compare, hp.model, hp.snr))
        num = main(hp, num)
