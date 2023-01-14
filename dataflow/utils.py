import yaml
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import pandas as pd
import os


def read_yaml(path):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config


def format_audio(current_path, self_samplerate, self_duration):
    data, samplerate = sf.read(current_path)
    # data = librosa.resample(data_raw, target_sr=self_samplerate, orig_sr=samplerate)
    duration = data.shape[0] / self_samplerate

    if duration > self_duration:
        data = data[:self_samplerate * self_duration]
    else:
        diff = self_duration * self_samplerate - data.shape[0]
        data = np.pad(data, pad_width=(0, diff))

    return data


def mp3_to_wav(languages, path):
    for language in languages:
        dataset = pd.read_table(os.path.join(path, language, 'validated.tsv'))
        for audio_path in tqdm(dataset['path']):
            out_path = audio_path[:-3]
            audio_to_convert = os.path.join(path, language, 'clips', audio_path)
            output_full_path = os.path.join(path, language, 'wav_clips', out_path)
            command = f"ffmpeg -i {audio_to_convert} -acodec pcm_s16le -ac 1 -ar 16000 {output_full_path}wav"
            os.system(command)
        # os.system(f'rm -r {language}')
        os.system(f'mv -T {os.path.join(path, language, "clips")} old_clips')
        os.system(f'rm -T {os.path.join(path, language, "wav_clips")} clips')
