import torch
import torch.utils.data
import random
import numpy as np
from .symbols import ph2id
import os
from torch.utils.data import DataLoader

def read_text_o(fn):
    text = []
    with open(fn) as f:
        lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            start, end, phone = line.strip().split()
            text.append([float(start), int(end), phone])
    return text

def read_text(fn):
    '''
    read phone alignments from file of the format:
    start end phone
    '''
    text = []
    with open(fn) as f:
        lines = f.readlines()
        lines = [l.strip().split() for l in lines]

    for i in lines:
        text.extend(i)
        #for line in lines:
            #start, end, phone = line.strip().split()
            #text.append([int(start), int(end), phone])
    return text

class TextMelIDLoader(torch.utils.data.Dataset):
    
    def __init__(self, list_file, mean_std_file, speaker_A, speaker_B,shuffle=True, pids=None):
        
        file_path_list = []

        with open(list_file) as f:
            lines = f.readlines()
            for line in lines:
                path, n_frame = line.strip().split()
                #speaker_id = path.split('/')[-3].split('_')[2]
                speaker_id = path.split('/')[-2]
                #path = path.replace('data07','home')
                #path = path.replace('CMU_ARCTIC','non')

                if not speaker_id in pids:
        
                    continue

                #if int(n_frame) >= 800:
                #    continue
                
                file_path_list.append(path)


        random.seed(1234)
        if shuffle:
            random.shuffle(file_path_list)
        
        self.file_path_list = file_path_list

        self.mel_mean_std = np.float32(np.load(mean_std_file))
        self.spc_mean_std = np.float32(np.load(mean_std_file.replace('mel', 'spec')))
        #self.sp2id = {speaker_A:0,speaker_B:1}
        #self.sp2id = {'slt': 0, 'rms': 1}
        self.sp2id={'Neutral':0,'Happy':1,'Sad':2,'Angry':3,'Surprise':4}

    
    def get_path_id(self, path):
        # Custom this function to obtain paths and speaker id
        # Deduce filenames
        text_path = path.replace('/CMU_ARCTIC', '').replace('/mel', '/txt').replace('.mel.npy', '.phones')
        b = text_path.split('/')[-1]
        text_path = os.path.join('/home/panzexu/kun/nonparaSeq2seqVC_code-master/0019/txt',b)

        mel_path = path.replace('spec', 'mel')
        #speaker_id = path.split('/')[-3].split('_')[2]
        speaker_id = path.split('/')[-2]
        # use non-trimed version #
        spec_path = path.replace('mel', 'spec')
        #text_path = text_path.replace('text_trim', 'text')
        #mel_path = mel_path.replace('mel_trim', 'mel')
        #speaker_id = path.split('/')[-3].split('_')[2]
        speaker_id = path.split('/')[-2]

        return mel_path, spec_path, text_path, speaker_id

    def get_text_mel_id_pair(self, path):
        '''
        text_input [len_text]
        text_targets [len_mel]
        mel [mel_bin, len_mel]
        speaker_id [1]
        '''

        mel_path, spec_path, text_path, speaker_id  = self.get_path_id(path)
        # Load data from disk
        text_input = self.get_text(text_path)
        mel = np.load(mel_path)
        spc = np.load(spec_path)
        speaker_id = [self.sp2id[speaker_id]]
        # Normalize audio 
        mel = (mel - self.mel_mean_std[0])/ self.mel_mean_std[1]
        spc = (spc - self.spc_mean_std[0]) / self.spc_mean_std[1]
        # Format for pytorch
        text_input = torch.LongTensor(text_input)
        mel = torch.from_numpy(np.transpose(mel))
        spc = torch.from_numpy(np.transpose(spc))
        speaker_id = torch.LongTensor(speaker_id)

        strength_angry = np.float32(1)
        strength_happy =  np.float32(0)
        strength_sad = np.float32(0)
        strength_neutral = np.float32(1)
        strength_surprise = np.float32(1)

        strength = np.array([strength_neutral, strength_happy, strength_sad, strength_angry, strength_surprise])
        strength = torch.tensor(strength)
        strength = torch.tensor(strength) * torch.ones(1)

        return (text_input, mel, spc, speaker_id, strength)
        
    def get_text(self,text_path):

        text = read_text(text_path)
        text_input = []

        #for start, end, ph in text:
            ##dur = int((end - start) / 125000. + 0.6)
            #text_input.append(ph2id[ph])
        for ph in text:

            text_input.append(ph2id[ph])

        return text_input

    def __getitem__(self, index):
        return self.get_text_mel_id_pair(self.file_path_list[index])

    def __len__(self):
        return len(self.file_path_list)


class TextMelIDCollate():

    def __init__(self, n_frames_per_step=2):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        '''
        batch is list of (text_input, mel, spc, speaker_id)
        '''

        text_lengths = torch.IntTensor([len(x[0]) for x in batch])
        mel_lengths = torch.IntTensor([x[1].size(1) for x in batch])
        mel_bin = batch[0][1].size(0)
        spc_bin = batch[0][2].size(0)

        max_text_len = torch.max(text_lengths).item()
        max_mel_len = torch.max(mel_lengths).item()
        if max_mel_len % self.n_frames_per_step != 0:
            max_mel_len += self.n_frames_per_step - max_mel_len % self.n_frames_per_step
            assert max_mel_len % self.n_frames_per_step == 0

        text_input_padded = torch.LongTensor(len(batch), max_text_len)
        mel_padded = torch.FloatTensor(len(batch), mel_bin, max_mel_len)
        spc_padded = torch.FloatTensor(len(batch), spc_bin, max_mel_len)

        speaker_id = torch.LongTensor(len(batch))
        stop_token_padded = torch.FloatTensor(len(batch), max_mel_len)

        strength_padded = torch.FloatTensor(len(batch), 5)

        text_input_padded.zero_()
        mel_padded.zero_()
        spc_padded.zero_()
        speaker_id.zero_()
        stop_token_padded.zero_()
        strength_padded.zero_()

        for i in range(len(batch)):
            text =  batch[i][0]
            mel = batch[i][1]
            spc = batch[i][2]

            text_input_padded[i,:text.size(0)] = text 
            mel_padded[i,  :, :mel.size(1)] = mel
            spc_padded[i,  :, :spc.size(1)] = spc
            speaker_id[i] = batch[i][3][0]
            #make sure the downsampled stop_token_padded have the last eng flag 1. 
            stop_token_padded[i, mel.size(1)-self.n_frames_per_step:] = 1 
            strength = batch[i][4]
            strength_padded[i] = strength * torch.ones(1)

        return text_input_padded, mel_padded, spc_padded, speaker_id, \
                    text_lengths, mel_lengths, stop_token_padded, strength_padded
