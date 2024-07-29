

import enum
import glob
import os
import random
from random import randint


def write_list():

    data_path = "I:\\datas\\voxceleb1\\*\\*\\*\\*\\*.wav"
    audios = glob.glob(data_path)

    total_audio_num = len(audios)
    print('total audio num:%d'%total_audio_num)

    speaker = os.listdir('I:\\datas\\voxceleb1\\train\\wav')
    print('total train speaker num:%d'%len(speaker))

    speakers = {}
    #生成说话人字典
    for i, audio in enumerate(audios):
        audio_path_list = audio.split('\\')
        name = audio_path_list[5]
        if name in speakers:
            speakers[name].append(audio) 
        else:
            speakers[name] = []
            speakers[name].append(audio)
    
    with open('.\\datas\\Vox1_train_list.txt', 'w') as f:
        for speaker in speakers.keys():
            for audio in speakers[speaker][:-1]:
                f.write(speaker + ' ' + audio + '\n')

    with open('.\\datas\\Vox1_test_list.txt', 'w') as f:
        for speaker in speakers.keys():
            f.write(speaker + ' ' + speakers[speaker][-1] + '\n')

def write_list_long():
    data_path = "I:\\datas\\voxceleb1\\train\\wav\\*\\*\\*.wav"
    audios = glob.glob(data_path)

    speakers = {}
    #生成说话人字典
    for i, audio in enumerate(audios):
        audio_path_list = audio.split('\\')
        name = audio_path_list[5]
        if name in speakers:
            speakers[name].append(audio) 
        else:
            speakers[name] = []
            speakers[name].append(audio)

    # with open('.\\datas/Vox1_rta_list.txt', 'w') as f:
    #     for speaker in speakers.keys():
    #         f.write(speaker + ' ' + speakers[speaker][-1] + ' ' + speakers[speaker][-2]  + ' ' + speakers[speaker][-3] + ' ' + speakers[speaker][-4] + '\n')

def write_list_sv():

    speaker = os.listdir('I:\\datas\\voxceleb1\\test\\wav')
    print('total test speaker num:%d'%len(speaker))

    data_path = 'I:\\datas\\voxceleb1\\test\\wav\\*\\*\\*.wav'
    audios = glob.glob(data_path)

    total_audio_num = len(audios)
    print('total test audio num:%d'%total_audio_num)

    speakers = {}
    #生成说话人字典
    for i, audio in enumerate(audios):
        audio_path_list = audio.split('\\')
        name = audio_path_list[5]
        if name in speakers:
            speakers[name].append(audio) 
        else:
            speakers[name] = []
            speakers[name].append(audio)

    with open('.\\datas\\Voxceleb_test_sv_list.txt', 'w') as f:
        for i, audio in enumerate(audios):
            audio_path_list = audio.split('\\')
            name = audio_path_list[5]
            neg_speaker = speaker.copy()
            neg_speaker.remove(name)
            neg = random.choices(neg_speaker, k=1)[0]
            neg = random.choices(speakers[neg], k=1)[0]
            pos = random.choices(speakers[name], k=1)[0]
            #audio = audio[36:]
            #pos = pos[36:]
            #neg = neg[36:]
            f.write('1' + ' ' + audio + ' ' + pos + '\n')
            f.write('0' + ' ' + audio + ' ' + neg + '\n')

if __name__ == "__main__":
    write_list()
    write_list_sv()