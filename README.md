# Mixed_Emotions
This is the code for "Speech Synthesis with Mixed Emotions", which we submitted to IEEE Transactions on Affective Computing. The Arxiv version is here: https://arxiv.org/abs/2208.05890.

## Brief Introduction
In this repo, we aims to synthesize mixed emotional effects for text-to-speech. The training procedure is similar with that of https://arxiv.org/abs/2103.16809, which consists of two stages: (1) Style Pre-training with a Multi-speaker TTS Corpus (VCTK dataset); (2) Emotion Adaptation with emotional speech data (ESD dataset). The first stage is time-consuming, thus I provide the pre-trained models for step I here: https://drive.google.com/file/d/1hRa-dygp1kBdp2IPKMPhGRdv9x5MT9o4/view?usp=sharing. You can directly download the Stage I model and perform Stage II. This repo starts from the step II. For more details about step I, please refer to https://github.com/KunZhou9646/seq2seq-EVC.

Pretrained Models:

1/ Stage I: https://drive.google.com/file/d/1hRa-dygp1kBdp2IPKMPhGRdv9x5MT9o4/view?usp=sharing

2/ Stage II: https://drive.google.com/file/d/1Btaw31axF1X5cwsqoK6OI40-Gz7PY5AB/view?usp=sharing


**1. Installation**
```Bash
$ pip install -r requirements.txt
```
**2. Pre-processing for Stage II: Emotion Training**

You need to download ESD corpus and customize it accordingly, and then perform feature extraction:
```Bash
$ cd emotion_adaptation
$ cd reader
$ python extract.py (please customize "path" and "kind", and edit the codes for "spec" or "mel-spec")
$ python generate_list_mel.py
```

**3. Stage II: Emotion Training**
```Bash
$ cd emotion_adaptation
$ python train.py -l logdir \
-o outdir --n_gpus=1 -c '/home/panzexu/kun/Mixed_Emotions/checkpoint_234000'[PATH TO STAGE-I PRETRAINED MODELS] --warm_start
```
**4. Run-time Inference**

(1) Generate emotion embedding from the emotion encoder:
```Bash
$ cd synthesis
$ python inference_embedding.py -c '/home/panzexu/kun/Mixed_Emotions/pre-train_IS_0019/outdir_new/checkpoint_11200'[PATH TO STAGE-II PRETRAINED MODELS] --hparams speaker_A='Neutral',speaker_B='Happy',speaker_C='Sad',speaker_D='Angry',speaker_E='Surprise',training_list='/home/panzexu/kun/Mixed_Emotions/synthesis/reader/emotion_list_0019/testing_mel_list.txt',SC_kernel_size=1
```

(2) Synthesize a mixture of emotions:
```Bash
$ cd conversion
$ python inference.py -c '/home/panzexu/kun/Mixed_Emotions/emotion_adaptation/outdir/checkpoint_11200'[PATH TO STAGE-II PRETRAINED MODELS] --num 20 --hparams validation_list='/home/panzexu/kun/Mixed_Emotions/synthesis/reader/emotion_list_0019/evaluation_mel_list.txt',SC_kernel_size=1
```

Please customize './reader/reader.py' to adjust the percentage of each emotions in the mixture. [line 116 - line 120]
