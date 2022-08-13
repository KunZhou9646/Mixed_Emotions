#!/bin/bash


#SBATCH --gres=gpu:1
#SBATCH --job-name=inference
#SBATCH -o res_emb.out 

source /home/zhoukun/miniconda3/bin/activate new

python inference_embedding.py -c '/home/panzexu/kun/nonparaSeq2seqVC_code-master/pre-train_IS_0019/outdir_new/checkpoint_44900' --hparams speaker_A='Neutral',speaker_B='Happy',speaker_C='Sad',speaker_D='Angry',speaker_E='Surprise',training_list='/home/panzexu/kun/nonparaSeq2seqVC_code-master/fine-tune-21/reader/emotion_list_0019/testing_mel_list.txt',SC_kernel_size=1
