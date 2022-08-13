# Mixed_Emotions
This is the code for "Speech Synthesis with Mixed Emotions", which we submitted to IEEE Transactions on Affective Computing. The Arxiv version is here: https://arxiv.org/abs/2208.05890.

## Brief Introduction
In this repo, we aims to synthesize mixed emotional effects for text-to-speech. The training procedure is similar with that of https://arxiv.org/abs/2103.16809, which consists of two stages: (1) Style Pre-training with a Multi-speaker TTS Corpus (VCTK dataset); (2) Emotion Adaptation with emotional speech data (ESD dataset). The first stage is time-consuming, thus I provide the pre-trained models for step I here: https://drive.google.com/file/d/1hRa-dygp1kBdp2IPKMPhGRdv9x5MT9o4/view?usp=sharing. You can directly download the Stage I model and perform Stage II. This repo starts from the step II. For more details about step I, please refer to https://github.com/KunZhou9646/seq2seq-EVC.

Note: I have tested this repo and I swear that it works with NO BUGs.

Pretrained Models:
1/ Stage I: https://drive.google.com/file/d/1hRa-dygp1kBdp2IPKMPhGRdv9x5MT9o4/view?usp=sharing
2/ Stage II: https://drive.google.com/file/d/1Btaw31axF1X5cwsqoK6OI40-Gz7PY5AB/view?usp=sharing


**1. Installation**
```Bash
$ pip install -r requirements.txt
```

