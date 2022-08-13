#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=train
#SBATCH -o res.out 

source /home/zhoukun/miniconda3/bin/activate seq2seq2


# slt to rms, rms to slt #
RUN_EMB=False
RUN_TRAIN=true
RUN_GEN=False

speaker_A='slt'
speaker_B='rms'

training_list = '/home/zhoukun/nonparaSeq2seqVC_code-master/fine-tune/reader/training_mel_list.txt'
validation_list = '/home/zhoukun/nonparaSeq2seqVC_code-master/fine-tune/reader/evaluation_mel_list.txt'


logdir="logdir_${speaker_A}_${speaker_B}"
pretrain_checkpoint_path='../pre-train/outdir/checkpoint_234000'
finetune_ckpt="checkpoint_100"

contrastive_loss_w=30.0
speaker_adversial_loss_w=0.2
speaker_classifier_loss_w=1.0
decay_every=7
warmup=7
epochs=70
batch_size=8
SC_kernel_size=1
learning_rate=1e-3
gen_num=66

if $RUN_EMB
then
    echo 'running embeddings...'
    python inference_embedding.py \ 
      -c $pretrain_checkpoint_path \
    --hparams=speaker_A=$speaker_A,\
speaker_B=$speaker_B,\
training_list=${training_list},SC_kernel_size=$SC_kernel_size
   
fi

if $RUN_TRAIN
then
    echo 'running trainings...'
    python train.py  \
        -l $logdir -o outdir --n_gpus=1 \
        -c $pretrain_checkpoint_path \
        --warm_start \
        --hparams=speaker_A=$speaker_A,\
speaker_B=$speaker_B,a_embedding_path="outdir/embeddings/${speaker_A}.npy",\
b_embedding_path="outdir/embeddings/${speaker_B}.npy",\
training_list=$training_list,\
validation_list=$validation_list,\
contrastive_loss_w=$contrastive_loss_w,\
speaker_adversial_loss_w=$speaker_adversial_loss_w,\
speaker_classifier_loss_w=$speaker_classifier_loss_w,\
decay_every=$decay_every,\
epochs=$epochs,\
warmup=$warmup,batch_size=$batch_size,\
SC_kernel_size=$SC_kernel_size,learning_rate=$learning_rate
fi


if $RUN_GEN
then 
    echo 'running generations...'
    python inference.py \
        -c outdir/$logdir/$finetune_ckpt \
        --num $gen_num \
        --hparams=validation_list=$validation_list,SC_kernel_size=$SC_kernel_size
fi
