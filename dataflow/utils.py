import csv
import yaml
import librosa
import soundfile as sf
import numpy as np


def read_yaml(path):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config


def format_audio(current_path, self_samplerate, self_duration):
    data_raw, samplerate = sf.read(current_path)
    data = librosa.resample(data_raw, target_sr=self_samplerate, orig_sr=samplerate)
    duration = data.shape[0] / self_samplerate

    if duration > self_duration:
        data = data[:self_samplerate * self_duration]
    else:
        diff = self_duration * self_samplerate - data.shape[0]
        data = np.pad(data, pad_width=(0, diff))

    return data

def read_tsv(path):
    with open("validated.tsv", 'r') as file:
        tsv_file = csv.reader(file, delimiter="\t")
        i = 0
        for line in tsv_file:
            i += 1
            print(line)
            if i == 2:
                break
