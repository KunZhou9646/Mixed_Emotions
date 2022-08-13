import librosa
import numpy as np
import glob
import os
from multiprocessing import Pool, cpu_count
import sys
    
def extract_mel_spec(filename):
    '''
    extract and save both log-linear and log-Mel spectrograms.
    saved spec shape [n_frames, 1025]
    saved mel shape [n_frames, 80]
    '''
    y, sample_rate = librosa.load(filename,sr=16000)
    #y, _ = librosa.effects.trim(y,top_db=20)

    spec = librosa.core.stft(y=y, 
                             n_fft=2048, 
                             hop_length=200, 
                             win_length=800,
                             window='hann',
                             center=True,
                             pad_mode='reflect')
    spec= librosa.magphase(spec)[0]
    log_spectrogram = np.log(spec).astype(np.float32)

    mel_spectrogram = librosa.feature.melspectrogram(S=spec, 
                                                     sr=sample_rate, 
                                                     n_mels=80,
                                                     power=1.0, #actually not used given "S=spec"
                                                     fmin=0.0,
                                                     fmax=None,
                                                     htk=False,
                                                     norm=1
                                                     )
    log_mel_spectrogram = np.log(mel_spectrogram).astype(np.float32)

    #filename = filename.replace('/data06_2/','/data07/zhoukun/')
    #file_test = filename[:-17]

    file_test = filename.replace('/wav','/mel') # /mel or /spec?
    file_test_dir = file_test[:-16]
    if not os.path.isdir(file_test_dir):

        os.makedirs(file_test_dir)

    #filename_1 = filename.replace(".wav", ".spec")

    #np.save(file=filename_1, arr=log_spectrogram.T)

    #filename_2 = filename.replace(".wav", ".mel")

    #np.save(file=filename_2, arr=log_mel_spectrogram.T)

    filename_1 = file_test.replace('.wav', '.mel')
    np.save(file=filename_1,arr=log_mel_spectrogram.T)
    #filename_1 = file_test.replace('.wav', '.spec')
    #np.save(file=filename_1,arr=log_spectrogram.T)

def extract_phonemes(filename):
    from phonemizer.phonemize import phonemize

    from phonemizer.backend import FestivalBackend

    from phonemizer.separator import Separator
    #FestivalBackend.set_festival_path("/home/zhoukun/merlin/tools/festival/src/main/festival")    
    with open(filename) as f:
        text=f.read()
        phones = phonemize(text, language='en-us', backend='festival', separator=Separator(phone=' ', syllable='', word=''))
    #filename = filename.replace('/data06_2/', '/data07/zhoukun/')
    #file_test = filename[:-16]
    filename = filename.replace('/text','/phones')
    file_test = filename[:-16]
    if not os.path.isdir(file_test):
        os.mkdir(file_test)
    with open(filename.replace(".txt", ".phones"), "w") as outfile:
        print(phones, file=outfile)

def extract_dir(root, kind):
    if kind =="audio":
        extraction_function=extract_mel_spec
        ext=".wav"
    elif kind =="text":
        extraction_function=extract_phonemes
        ext=".txt"
    else:
        print("ERROR: invalid args")
        sys.exit(1)
    if not os.path.isdir(root):
        print("ERROR: invalid args")
        sys.exit(1)
        
    # traverse over all subdirs of the provided dir, and find
    # only files with the proper extension
    abs_paths=[]
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            abs_path = os.path.abspath(os.path.join(dirpath, f))
            if abs_path.endswith(ext):
                 abs_paths.append(abs_path)
    print(abs_paths)
    pool = Pool(cpu_count())
    pool.map(extraction_function,abs_paths)

    #estimate and save mean std statistics in root dir.
    #root = root.replace('/data06_2/', '/data07/zhoukun/')
    #root = root.replace('wav','mel')
    root = root.replace('wav', 'mel')
    if not os.path.exists(root):
        os.makedirs(root)
    estimate_mean_std(root)


def estimate_mean_std(root, num=2000):
    '''
    use the training data for estimating mean and standard deviation
    use $num utterances to avoid out of memory
    '''
    specs, mels = [], []
    counter_sp, counter_mel = 0, 0
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith('.spec.npy') and counter_sp<num:
                path = os.path.join(dirpath, f)
                specs.append(np.load(path))
                counter_sp += 1
            if f.endswith('.mel.npy') and counter_mel<num:
                path = os.path.join(dirpath, f)
                mels.append(np.load(path))
                counter_mel += 1
    
    #specs = np.vstack(specs)
    mels = np.vstack(mels)

    mel_mean = np.mean(mels,axis=0)
    mel_std = np.std(mels, axis=0)
    #spec_mean = np.mean(specs, axis=0)
    #spec_std = np.std(specs, axis=0)

    #np.save(os.path.join(root,"spec_mean_std.npy"),
        #[spec_mean, spec_std])



    np.save(os.path.join(root,"mel_mean_std.npy"),[mel_mean, mel_std])

        
if __name__ == "__main__":
    
    #path = '/data06_2/CMU_ARCTIC/cmu_us_awb_arctic/wav'
    path = '/home/zhoukun/nonparaSeq2seqVC_code-master/0013/wav'
    kind = 'audio'

    extract_dir(path,kind)
    
