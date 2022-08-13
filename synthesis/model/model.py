import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from math import sqrt
from .utils import to_gpu
from .decoder import Decoder
from .layers import SpeakerClassifier, SpeakerEncoder, AudioSeq2seq, TextEncoder,  PostNet, MergeNet
import os
from .basic_layers import LinearNorm

class Parrot(nn.Module):
    def __init__(self, hparams):
        super(Parrot, self).__init__()

        #print hparams
        # plus <sos> 
        self.embedding = nn.Embedding(
            hparams.n_symbols + 1, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std

        self.sos = hparams.n_symbols

        self.embedding.weight.data.uniform_(-val, val)

        self.text_encoder = TextEncoder(hparams)

        self.audio_seq2seq = AudioSeq2seq(hparams)

        self.merge_net = MergeNet(hparams)

        self.speaker_encoder = SpeakerEncoder(hparams)

        self.speaker_classifier = SpeakerClassifier(hparams)

        self.decoder = Decoder(hparams)
        
        self.postnet = PostNet(hparams)

        self._initilize_emb(hparams)

        self.spemb_input = hparams.spemb_input

        self.strength_projection = LinearNorm(5, 64)

    def _initilize_emb(self, hparams):


        # for directory in [hparams.a_embedding_path,hparams.b_embedding_path,hparams.c_embedding_path,hparams.d_embedding_path]:
        #     if not os.path.isfile(directory):
        #         f = open(directory,'a+')
        #         f.close()

        a_embedding = np.load(hparams.a_embedding_path)
        a_embedding = np.mean(a_embedding, axis=0)
        
        b_embedding = np.load(hparams.b_embedding_path)
        b_embedding = np.mean(b_embedding, axis=0)

        c_embedding = np.load(hparams.c_embedding_path)
        c_embedding = np.mean(c_embedding, axis=0)
        
        d_embedding = np.load(hparams.d_embedding_path)
        d_embedding = np.mean(d_embedding, axis=0)

        e_embedding = np.load(hparams.e_embedding_path)
        e_embedding = np.mean(e_embedding, axis=0)
        
        self.sp_embedding = nn.Embedding(
            hparams.n_speakers, hparams.speaker_embedding_dim)

        self.sp_embedding.weight.data[0] =  torch.FloatTensor(a_embedding) 
        self.sp_embedding.weight.data[1] =  torch.FloatTensor(b_embedding)
        self.sp_embedding.weight.data[2] =  torch.FloatTensor(c_embedding) 
        self.sp_embedding.weight.data[3] =  torch.FloatTensor(d_embedding)
        self.sp_embedding.weight.data[4] =  torch.FloatTensor(e_embedding)

    def grouped_parameters(self,):

        params_group1 = [p for p in self.embedding.parameters()]
        params_group1.extend([p for p in self.text_encoder.parameters()])
        params_group1.extend([p for p in self.audio_seq2seq.parameters()])

        params_group1.extend([p for p in self.sp_embedding.parameters()])
        params_group1.extend([p for p in self.merge_net.parameters()])
        params_group1.extend([p for p in self.decoder.parameters()])
        params_group1.extend([p for p in self.postnet.parameters()])

        return params_group1, [p for p in self.speaker_classifier.parameters()]

    def parse_batch(self, batch):
        text_input_padded, mel_padded, spc_padded, speaker_id, \
                    text_lengths, mel_lengths, stop_token_padded, strength_embedding = batch
        
        text_input_padded = to_gpu(text_input_padded).long()
        mel_padded = to_gpu(mel_padded).float()
        spc_padded = to_gpu(spc_padded).float()
        speaker_id = to_gpu(speaker_id).long()
        text_lengths = to_gpu(text_lengths).long()
        mel_lengths = to_gpu(mel_lengths).long()
        stop_token_padded = to_gpu(stop_token_padded).float()
        strength_embedding = to_gpu(strength_embedding).float()

        return ((text_input_padded, mel_padded, text_lengths, mel_lengths, speaker_id,strength_embedding),
                (text_input_padded, mel_padded, spc_padded,  speaker_id, stop_token_padded))


    def forward(self, inputs, input_text):

        text_input_padded, mel_padded, text_lengths, mel_lengths, speaker_id, strength_embedding = inputs

        text_input_embedded = self.embedding(text_input_padded.long()).transpose(1, 2) # -> [B, text_embedding_dim, max_text_len]
        text_hidden = self.text_encoder(text_input_embedded, text_lengths) # -> [B, max_text_len, hidden_dim]

        B = text_input_padded.size(0)
        start_embedding = Variable(text_input_padded.data.new(B,).fill_(self.sos))
        start_embedding = self.embedding(start_embedding)

        speaker_embedding = self.sp_embedding(speaker_id)

        if self.spemb_input:
            T = mel_padded.size(2)
            audio_input = torch.cat((mel_padded,
                speaker_embedding.detach().unsqueeze(2).expand(-1, -1, T)), dim=1)
        else:
            audio_input = mel_padded

        #-> [B, text_len+1, hidden_dim] [B, text_len+1, n_symbols] [B, text_len+1, T/r]
        audio_seq2seq_hidden, audio_seq2seq_logit, audio_seq2seq_alignments = self.audio_seq2seq(
                audio_input, mel_lengths, text_input_embedded, start_embedding)
        audio_seq2seq_hidden= audio_seq2seq_hidden[:,:-1, :] # -> [B, text_len, hidden_dim]

        speaker_logit_from_mel_hidden = self.speaker_classifier(audio_seq2seq_hidden) # -> [B, text_len, n_speakers]

        if input_text:
            hidden = self.merge_net(text_hidden, text_lengths)
        else:
            hidden = self.merge_net(audio_seq2seq_hidden, text_lengths)

        L = hidden.size(1)
        
        strength_embedding = self.strength_projection(strength_embedding)

        output = torch.cat([speaker_embedding,strength_embedding],-1)
        hidden = torch.cat([hidden, output.unsqueeze(1).expand(-1, L, -1)], -1)

        predicted_mel, predicted_stop, alignments = self.decoder(hidden, mel_padded, text_lengths)

        post_output = self.postnet(predicted_mel)

        outputs = [predicted_mel, post_output, predicted_stop, alignments,
                  text_hidden, audio_seq2seq_hidden, audio_seq2seq_logit, audio_seq2seq_alignments,
                  speaker_logit_from_mel_hidden,
                  text_lengths, mel_lengths]

        return outputs


    def inference(self, inputs, input_text, id_reference, beam_width):

        text_input_padded, mel_padded, text_lengths, mel_lengths, speaker_id, strength_embedding = inputs
        text_input_embedded = self.embedding(text_input_padded.long()).transpose(1, 2)
        text_hidden = self.text_encoder.inference(text_input_embedded)

        B = text_input_padded.size(0) # B should be 1
        start_embedding = Variable(text_input_padded.data.new(B,).fill_(self.sos))
        start_embedding = self.embedding(start_embedding) # [1, embedding_dim]

        #-> [B, text_len+1, hidden_dim] [B, text_len+1, n_symbols] [B, text_len+1, T/r]
       
        speaker_embedding = self.sp_embedding(speaker_id)

        if self.spemb_input:
            T = mel_padded.size(2)
            audio_input = torch.cat((mel_padded, 
                speaker_embedding.unsqueeze(2).expand(-1,-1,T)), dim=1)
        else:
            audio_input = mel_padded
        
        audio_seq2seq_hidden, audio_seq2seq_phids, audio_seq2seq_alignments = self.audio_seq2seq.inference_beam(
                audio_input, start_embedding, self.embedding, beam_width=beam_width) 
        audio_seq2seq_hidden= audio_seq2seq_hidden[:,:-1, :] # -> [B, text_len, hidden_dim]

        speaker_embedding = self.sp_embedding(id_reference)

        if input_text:
            hidden = self.merge_net.inference(text_hidden)
        else:
            hidden = self.merge_net.inference(audio_seq2seq_hidden)

        L = hidden.size(1)

        strength_embedding = self.strength_projection(strength_embedding)

        output = torch.cat([speaker_embedding,strength_embedding],-1)

        hidden = torch.cat([hidden, output.unsqueeze(1).expand(-1, L, -1)], -1)
          
        predicted_mel, predicted_stop, alignments = self.decoder.inference(hidden)

        post_output = self.postnet(predicted_mel)

        return (predicted_mel, post_output, predicted_stop, alignments,
            text_hidden, audio_seq2seq_hidden, audio_seq2seq_phids, audio_seq2seq_alignments,
            speaker_id)


