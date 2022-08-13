import torch
from torch import nn
from torch.nn import functional as F
from .utils import get_mask_from_lengths

class ParrotLoss(nn.Module):
    def __init__(self, hparams):
        super(ParrotLoss, self).__init__()
        
        self.L1Loss = nn.L1Loss(reduction='none')
        self.MSELoss = nn.MSELoss(reduction='none')
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='none')
        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')
        self.n_frames_per_step = hparams.n_frames_per_step_decoder
        self.eos = hparams.n_symbols
        self.predict_spectrogram = hparams.predict_spectrogram

        self.contr_w = hparams.contrastive_loss_w
        self.spenc_w = hparams.speaker_encoder_loss_w
        self.texcl_w = hparams.text_classifier_loss_w
        self.spadv_w = hparams.speaker_adversial_loss_w
        self.spcla_w = hparams.speaker_classifier_loss_w

        self.speaker_A = hparams.speaker_A
        self.speaker_B = hparams.speaker_B

    def parse_targets(self, targets, text_lengths):
        '''
        text_target [batch_size, text_len]
        mel_target [batch_size, mel_bins, T]
        spc_target [batch_size, spc_bins, T]
        speaker_target [batch_size]
        stop_target [batch_size, T]
        '''
        text_target, mel_target, spc_target, speaker_target, stop_target = targets

        B = stop_target.size(0)
        stop_target = stop_target.reshape(B, -1, self.n_frames_per_step)
        stop_target = stop_target[:, :, 0]

        padded = torch.tensor(text_target.data.new(B,1).zero_())
        text_target = torch.cat((text_target, padded), dim=-1)
        
        # adding the ending token for target
        for bid in range(B):
            text_target[bid, text_lengths[bid].item()] = self.eos

        return text_target, mel_target, spc_target, speaker_target, stop_target
    
    def forward(self, model_outputs, targets, eps=1e-5):

        '''
        predicted_mel [batch_size, mel_bins, T]
        predicted_stop [batch_size, T/r]
        alignment 
            when input_text==True [batch_size, T/r, max_text_len] 
            when input_text==False [batch_size, T/r, T/r]
        text_hidden [B, max_text_len, hidden_dim]
        mel_hidden [B, max_text_len, hidden_dim]
        text_logit_from_mel_hidden [B, max_text_len+1, n_symbols+1]
        speaker_logit_from_mel [B, n_speakers]
        speaker_logit_from_mel_hidden [B, max_text_len, n_speakers]
        text_lengths [B,]
        mel_lengths [B,]
        '''
        predicted_mel, post_output, predicted_stop, alignments,\
            text_hidden, mel_hidden, text_logit_from_mel_hidden, \
            audio_seq2seq_alignments, \
             speaker_logit_from_mel_hidden, \
             text_lengths, mel_lengths = model_outputs

        text_target, mel_target, spc_target, speaker_target, stop_target  = self.parse_targets(targets, text_lengths)

        
        ## get masks ##
        mel_mask = get_mask_from_lengths(mel_lengths, mel_target.size(2)).unsqueeze(1).expand(-1, mel_target.size(1), -1).float()
        spc_mask = get_mask_from_lengths(mel_lengths, mel_target.size(2)).unsqueeze(1).expand(-1, spc_target.size(1), -1).float()

        mel_step_lengths = torch.ceil(mel_lengths.float() / self.n_frames_per_step).long()
        stop_mask = get_mask_from_lengths(mel_step_lengths, 
                                    int(mel_target.size(2)/self.n_frames_per_step)).float() # [B, T/r]
        text_mask = get_mask_from_lengths(text_lengths).float()
        text_mask_plus_one = get_mask_from_lengths(text_lengths + 1).float()

        # reconstruction loss #
        recon_loss = torch.sum(self.L1Loss(predicted_mel, mel_target) * mel_mask) / torch.sum(mel_mask)

        if self.predict_spectrogram:
            recon_loss_post = (self.L1Loss(post_output, spc_target) * spc_mask).sum() / spc_mask.sum()
        else:
            recon_loss_post = (self.L1Loss(post_output, mel_target) * mel_mask).sum() / torch.sum(mel_mask)
        
        stop_loss = torch.sum(self.BCEWithLogitsLoss(predicted_stop, stop_target) * stop_mask) / torch.sum(stop_mask)


        if self.contr_w == 0.:
            contrast_loss = torch.tensor(0.).cuda()
        else:
            # contrastive mask #
            contrast_mask1 =  get_mask_from_lengths(text_lengths).unsqueeze(2).expand(-1, -1, mel_hidden.size(1)) # [B, text_len] -> [B, text_len, T/r]
            contrast_mask2 = get_mask_from_lengths(text_lengths).unsqueeze(1).expand(-1, text_hidden.size(1), -1) # [B, T/r] -> [B, text_len, T/r]
            contrast_mask = (contrast_mask1 & contrast_mask2).float() 

            text_hidden_normed = text_hidden / (torch.norm(text_hidden, dim=2, keepdim=True) + eps)
            mel_hidden_normed = mel_hidden / (torch.norm(mel_hidden, dim=2, keepdim=True) + eps)

            # (x - y) ** 2 = x ** 2 + y ** 2 - 2xy
            distance_matrix_xx = torch.sum(text_hidden_normed ** 2, dim=2, keepdim=True) #[batch_size, text_len, 1]
            distance_matrix_yy = torch.sum(mel_hidden_normed ** 2, dim=2)
            distance_matrix_yy = distance_matrix_yy.unsqueeze(1) #[batch_size, 1, text_len]

            #[batch_size, text_len, text_len]
            distance_matrix_xy = torch.bmm(text_hidden_normed, torch.transpose(mel_hidden_normed, 1, 2)) 
            distance_matrix = distance_matrix_xx + distance_matrix_yy - 2 * distance_matrix_xy
            
            TTEXT = distance_matrix.size(1)
            hard_alignments = torch.eye(TTEXT).cuda()
            contrast_loss = hard_alignments * distance_matrix + \
                (1. - hard_alignments) * torch.max(1. - distance_matrix, torch.zeros_like(distance_matrix))

            contrast_loss = torch.sum(contrast_loss * contrast_mask) / torch.sum(contrast_mask)

        n_speakers = speaker_logit_from_mel_hidden.size(2)
        TTEXT = speaker_logit_from_mel_hidden.size(1)
        n_symbols_plus_one = text_logit_from_mel_hidden.size(2)

        speaker_encoder_loss = torch.tensor(0.).cuda()
        speaker_encoder_acc = torch.tensor(0.).cuda()

        
        speaker_logit_flatten = speaker_logit_from_mel_hidden.reshape(-1) # -> [B* TTEXT]
        predicted_speaker = (F.sigmoid(speaker_logit_flatten) > 0.5).long()
        speaker_target_flatten = speaker_target.unsqueeze(1).expand(-1, TTEXT).reshape(-1)

        speaker_classification_acc = ((predicted_speaker == speaker_target_flatten).float() * text_mask.reshape(-1)).sum() / text_mask.sum()
        loss = self.BCEWithLogitsLoss(speaker_logit_flatten, speaker_target_flatten.float())
       

        speaker_classification_loss = torch.sum(loss * text_mask.reshape(-1)) / torch.sum(text_mask)

        # text classification loss #
        text_logit_flatten = text_logit_from_mel_hidden.reshape(-1, n_symbols_plus_one)
        text_target_flatten = text_target.reshape(-1)
        _, predicted_text =  torch.max(text_logit_flatten, dim=1)
        text_classification_acc = ((predicted_text == text_target_flatten).float()*text_mask_plus_one.reshape(-1)).sum()/text_mask_plus_one.sum()
        loss = self.CrossEntropyLoss(text_logit_flatten, text_target_flatten)
        text_classification_loss = torch.sum(loss * text_mask_plus_one.reshape(-1)) / torch.sum(text_mask_plus_one)

        # speaker adversival loss #
        flatten_target = 0.5 * torch.ones_like(speaker_logit_flatten)
        loss = self.MSELoss(F.sigmoid(speaker_logit_flatten), flatten_target)
        mask = text_mask.reshape(-1)
        speaker_adversial_loss = torch.sum(loss * mask) / torch.sum(mask)

        loss_list = [recon_loss, recon_loss_post,  stop_loss,
                contrast_loss,  speaker_encoder_loss, speaker_classification_loss,
                text_classification_loss, speaker_adversial_loss]
            
        acc_list = [speaker_encoder_acc, speaker_classification_acc, text_classification_acc]
        
        combined_loss1 = recon_loss + recon_loss_post + stop_loss + self.contr_w * contrast_loss +  \
            self.texcl_w * text_classification_loss + \
            self.spadv_w * speaker_adversial_loss

        combined_loss2 = self.spcla_w * speaker_classification_loss
        


        return loss_list, acc_list, combined_loss1, combined_loss2


def torch_test_grad():
    
    x = torch.ones((1,1))

    net1 = nn.Linear(1, 1, bias=False)

   
    net1.weight.data.fill_(2.)
    net2 = nn.Linear(1, 1, bias=False)
    net2.weight.data.fill_(3.)

    all_params = []

    all_params.extend([p for p in net1.parameters()])
    all_params.extend([p for p in net2.parameters()])
    #print all_params

    y = net1(x) ** 2

    z = net2(y) ** 2

    loss1 = (z - 0.)
    loss2 = -5. * (z  - 0.) ** 2


    for p in net2.parameters():
        p.requires_grad = False

    loss1.backward(retain_graph=True)

    
    print((net1.weight.grad))
    print((net2.weight.grad))

    opt = torch.optim.SGD(all_params, lr=0.1)
    opt.step()

    print((net1.weight))
    print((net2.weight))
    #net1.weight.data = net1.weight.data - 0.1 * net1.weight.grad.data

    for p in net2.parameters():
        p.requires_grad=True
    
    for p in net1.parameters():
        p.requires_grad=False
    
    loss2.backward()
    print((net1.weight))
    print((net2.weight.grad))
    print((net1.weight.grad))
    
    net1.zero_grad()
    print((net1.weight.grad))

def test_logic():
    a = torch.ByteTensor([1,0,0,0,0])
    b = torch.ByteTensor([1,1,1,0,0])

    print(~a)
    print(a & b)
    print(a | b)

    text_lengths = torch.IntTensor([2,4,3]).cuda()
    mel_hidden_lengths =torch.IntTensor([5,6,5]).cuda()
    contrast_mask1 =  get_mask_from_lengths(text_lengths).unsqueeze(2).expand(-1, -1, 6) # [B, text_len] -> [B, text_len, T/r]
    contrast_mask2 = get_mask_from_lengths(mel_hidden_lengths).unsqueeze(1).expand(-1, 4, -1) # [B, T/r] -> [B, text_len, T/r]
    contrast_mask = contrast_mask1 & contrast_mask2 
    print(contrast_mask)
 
if __name__ == '__main__':

    torch_test_grad()
        
