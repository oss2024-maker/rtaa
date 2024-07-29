


# -*- coding: utf-8 -*-

import enum
import glob
import os
from random import randint

import tqdm
import subprocess


def concert():
    files = glob.glob('I:\\datas\\LibriSpeech\\*\\*\\*\\*\\*\\*.flac')
    files.sort()

    for fname in files:
        outfile = fname.replace('.flac', '.wav')
        os.system('ffmpeg -y -i %s -vn -acodec pcm_s16le -ar 16000 %s -loglevel quiet' %(fname, outfile))
        print('finish:%s' %fname)

def remv():
    files = glob.glob('I:\\datas\\LibriSpeech\\*\\*\\*\\*\\*\\*.flac')
    for fname in files:
        os.remove(fname)
        print('remove: %s' %fname)

def train_list():
    data_path = 'I:\\datas\\LibriSpeech\\train-clean-100\\LibriSpeech\\*\\*\\*\\*.wav'
    audios = glob.glob(data_path)

    total_speaker_num = len(audios)
    print('total train num:%d'%total_speaker_num)

    with open('.\\datas\\LibriSpeech_train_list.txt', 'w') as f:
        for i, audio in enumerate(audios):
            audio_path_list = audios[i].split('\\')
            f.write(audio_path_list[-3] + ' ' + audio + '\n')


def write_list():

    data_path = 'I:\\datas\\LibriSpeech\\train-clean-100\\LibriSpeech\\*\\*\\*\\*.wav'
    audios = glob.glob(data_path)

    total_audio_num = len(audios)
    print('total audio num:%d'%total_audio_num)

    speaker = os.listdir('I:\\datas\\Librispeech\\train-clean-100\\LibriSpeech\\train-clean-100')
    print('total train speaker num:%d'%len(speaker))

    speakers = {}
    #生成说话人字典
    for i, audio in enumerate(audios):
        audio_path_list = audio.split('\\')
        name = audio_path_list[6]
        if name in speakers:
            speakers[name].append(audio) 
        else:
            speakers[name] = []
            speakers[name].append(audio)
    
    with open('.\\datas\\Libri_train_list.txt', 'w') as f:
        for speaker in speakers.keys():
            for audio in speakers[speaker][:-1]:
                f.write(speaker + ' ' + audio + '\n')

    with open('.\\datas\\Libri_test_list.txt', 'w') as f:
        for speaker in speakers.keys():
            f.write(speaker + ' ' + speakers[speaker][-1] + '\n')

def write_list_long():

    data_path = 'I:\\datas\\LibriSpeech\\train-clean-100\\LibriSpeech\\*\\*\\*\\*.wav'
    audios = glob.glob(data_path)

    total_audio_num = len(audios)
    print('total audio num:%d'%total_audio_num)

    speaker = os.listdir('I:\\datas\\Librispeech\\train-clean-100\\LibriSpeech\\train-clean-100')
    print('total train speaker num:%d'%len(speaker))

    speakers = {}
    #生成说话人字典
    for i, audio in enumerate(audios):
        audio_path_list = audio.split('\\')
        name = audio_path_list[6]
        if name in speakers:
            speakers[name].append(audio) 
        else:
            speakers[name] = []
            speakers[name].append(audio)
    
    # with open('.\\datas\\Libri_train_list.txt', 'w') as f:
    #     for speaker in speakers.keys():
    #         for audio in speakers[speaker][:-1]:
    #             f.write(speaker + ' ' + audio + '\n')

    with open('.\\datas\\Libri_test_long_list.txt', 'w') as f:
        for speaker in speakers.keys():
            f.write(speaker + ' ' + speakers[speaker][-5] + ' ' + speakers[speaker][-4]  + ' ' + speakers[speaker][-3]  + ' ' + speakers[speaker][-2]  + ' ' + speakers[speaker][-1] +'\n')

def test_list():
    data_path = 'I:\\datas\\LibriSpeech\\test-clean\\LibriSpeech\\*\\*\\*\\*.wav'
    audios = glob.glob(data_path)

    total_speaker_num = len(audios)
    print('total test num:%d'%total_speaker_num)

    speaker_path = 'I:\\datas\\LibriSpeech\\test-clean\\LibriSpeech\\test-clean'
    speakers = os.listdir(speaker_path)
    print(speakers)

    pos_dict = {}
    nev_dict = {}
    with open('.\\datas\\LibriSpeech_test_list.txt', 'w') as f:
        for i, sp in enumerate(speakers):
            #print(sp)
            if i < 20:
                path_pos = speaker_path + '\\' + sp
                value_list = glob.glob('%s\\*\\*.wav'%path_pos)
                pos_dict[i] = value_list
                #print(value_list)
            else:
                print(sp)
                path_nev = speaker_path + '\\' + sp
                value_list = glob.glob('%s\\*\\*.wav'%path_nev)
                nev_dict[i] = value_list
                print(value_list)
        #print(pos_dict)
        #print(nev_dict)
        for i in range(20):
            for j in range(int(len(pos_dict[i])/2)):
                f.write('1' + ' ' + pos_dict[i][j] + ' ' + pos_dict[i][j+int(len(pos_dict)/2)] + '\n')
                rand_i = randint(0, 19)
                rand = randint(0, len(nev_dict[rand_i+20])-1)
                f.write('0' + ' ' + pos_dict[i][j] + ' ' + nev_dict[rand_i+20][rand] + '\n')



if __name__ == "__main__":
    #concert()
    #remv()
    #train_list()
    #write_list()
    #test_list()
    write_list_long()