import os
import sys
import random
import numpy as np
# 99 speakers
# for each speaker, 10 utterances for validation, 20 utterances for testing, the remaining for training 



def cal_phone_num(file):
    phone = open(file)
    line = phone.readline()
    num = len(line.split())
    return num

def cal_frame_num(file):
    sp = np.load(file)
    num = sp.shape[0]
    return num



train_list = []
eval_list = []
test_list = []

train_file_list = []
eval_file_list = []
test_file_list = []

#seen_speakers = ['p336', 'p240', 'p262', 'p333', 'p297', 'p339', 'p276', 'p269', 'p303', 'p260', 'p250', 'p345', 'p305', 'p283', 'p277', 'p302', 'p280', 'p295', 'p245', 'p227', 'p257', 'p282', 'p259', 'p311', 'p301', 'p265', 'p270', 'p329', 'p362', 'p343', 'p246', 'p247', 'p351', 'p263', 'p363', 'p249', 'p231', 'p292', 'p304', 'p347', 'p314', 'p244', 'p261', 'p298', 'p272', 'p308', 'p299', 'p234', 'p268', 'p271', 'p316', 'p287', 'p318', 'p264', 'p313', 'p236', 'p238', 'p334', 'p312', 'p230', 'p253', 'p323', 'p361', 'p275', 'p252', 'p374', 'p286', 'p274', 'p254', 'p310', 'p306', 'p294', 'p326', 'p225', 'p255', 'p293', 'p278', 'p266', 'p229', 'p335', 'p281', 'p307', 'p256', 'p243', 'p364', 'p239', 'p232', 'p258', 'p267', 'p317', 'p284', 'p300', 'p288', 'p341', 'p340', 'p279', 'p330', 'p360', 'p285']
seen_speakers = ['awb','bdl','clb','slt','rms']

i = 0
for speaker in seen_speakers:
    files=[]
    #files = [os.path.join('/data07/zhoukun/VCTK-Corpus/spec/%s'%speaker, fn) for fn in os.listdir('/data07/zhoukun/VCTK-Corpus/spec/%s'%speaker)]
    filesdir = os.path.join('/data07/zhoukun/CMU_ARCTIC/cmu_us_%s_arctic/spec' % speaker)
    for file in os.listdir(filesdir):
        file_path = os.path.join(filesdir,file)
        files.append(file_path)
    #random.shuffle(files)
    files.sort()
    eval_files = files[:66]
    test_files = files[66:132]
    train_files = files[132 + 500*i: 132 + 500*(i+1)]
    i = i + 1
    train_list.extend(train_files)
    eval_list.extend(eval_files)
    test_list.extend(test_files)

file_list = []
num_frame_list = []
num_phone_list = []

print('preparing evaluation list...')
for file in eval_list:

    num_frame = cal_frame_num(file)
    file_phone = file.replace('spec','txt')
    #file_phone = file_phone.replace('.txt.npy','.phones')
    #num_phone = cal_phone_num(file_phone)
    file_list.append(file)
    num_frame_list.append(num_frame)
    #num_phone_list.append(num_phone)

print('generating evaluation list ...')
f = open('evaluation_spec_list.txt','w')
for index in range(len(file_list)):
    f.write(str(file_list[index]) + ' ' + str(num_frame_list[index]) + '\n')
f.close()

file_list = []
num_frame_list = []
num_phone_list = []

print('preparing testing list...')
for file in test_list:

    num_frame = cal_frame_num(file)
    #file_phone = file.replace('spec','txt')
    #file_phone = file_phone.replace('.txt.npy','.phones')
    #num_phone = cal_phone_num(file_phone)
    file_list.append(file)
    num_frame_list.append(num_frame)
    #num_phone_list.append(num_phone)

print('generating testing list ...')
f = open('testing_spec_list.txt','w')
for index in range(len(file_list)):
    f.write(str(file_list[index]) + ' ' + str(num_frame_list[index]) + '\n')
f.close()
file_list = []
num_frame_list = []
num_phone_list = []

print('preparing training list...')
for file in train_list:

    num_frame = cal_frame_num(file)
    #file_phone = file.replace('spec','txt')
    #file_phone = file_phone.replace('.txt.npy','.phones')
    #num_phone = cal_phone_num(file_phone)
    #train_file_list = [file, num_frame,num_phone]
    #train_file_list.extend(train_file_list)
    file_list.append(file)
    num_frame_list.append(num_frame)
    #num_phone_list.append(num_phone)

print('generating training list ...')
f = open('training_spec_list.txt','w')
for index in range(len(file_list)):
    f.write(str(file_list[index]) + ' ' + str(num_frame_list[index])  + '\n')
f.close()



