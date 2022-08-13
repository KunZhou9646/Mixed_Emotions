#!/bin/bash
#SBATCH -w hlt02
#SBATCH --gres=gpu:1
#SBATCH --job-name=train
#SBATCH -o res_1.out 

source /home/zhoukun/miniconda3/bin/activate s2s

python train_src.py -l logdir_emotion_vc -o outdir --n_gpus=1 -c '/home/zhoukun/nonparaSeq2seqVC_code-master/pre-train/outdir/checkpoint_234000' --warm_start --hparams speaker_A=Neutral,speaker_B=Happy,speaker_C=Sad,speaker_D=Angry,speaker_E=Surprise,a_embedding_path='/home/zhoukun/nonparaSeq2seqVC_code-master/fine-tune/outdir/embeddings/Neutral.npy',b_embedding_path='/home/zhoukun/nonparaSeq2seqVC_code-master/fine-tune/outdir/embeddings/Happy.npy',c_embedding_path='/home/zhoukun/nonparaSeq2seqVC_code-master/fine-tune/outdir/embeddings/Sad.npy',d_embedding_path='/home/zhoukun/nonparaSeq2seqVC_code-master/fine-tune/outdir/embeddings/Angry.npy',e_embedding_path='/home/zhoukun/nonparaSeq2seqVC_code-master/fine-tune/outdir/embeddings/Surprise.npy',
training_list='/home/zhoukun/nonparaSeq2seqVC_code-master/fine-tune/reader/emotion_list/training_mel_list.txt',validation_list='/home/zhoukun/nonparaSeq2seqVC_code-master/fine-tune/reader/emotion_list/evaluation_mel_list.txt',contrastive_loss_w=30.0,speaker_adversial_loss_w=0.2,speaker_classifier_loss_w=1.0,decay_every=7,epochs=70,warmup=7,batch_size=8,SC_kernel_size=1,learning_rate=1e-3
