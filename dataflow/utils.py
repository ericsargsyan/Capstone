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


def format_audio(current_path, self_samplerate, self_duration, resample):
    if resample:
        data_raw, samplerate = sf.read(current_path)
        data = librosa.resample(data_raw, target_sr=self_samplerate, orig_sr=samplerate)
    else:
        data, samplerate = sf.read(current_path)

    duration = data.shape[0] / self_samplerate

    if duration > self_duration:
        data = data[:self_samplerate * self_duration]
    else:
        diff = self_duration * self_samplerate - data.shape[0]

        probability = np.random.rand()
        p, p_1 = int(diff * probability), diff - int(diff * probability)
        data = np.pad(data, pad_width=(p, p_1))

    return data


def mp3_to_wav(languages, path):
    for language in languages:
        dataset = pd.read_table(os.path.join(path, language, 'validated.tsv'))
        os.makedirs(os.path.join(path, language, 'wav_clips'), exist_ok=True)
        partial_path = os.path.join(path, language)

        for audio_path in tqdm(dataset['path']):
            out_path = audio_path[:-3]
            audio_to_convert = os.path.join(path, language, 'clips', audio_path)
            output_full_path = os.path.join(path, language, 'wav_clips', out_path)
            command = f"ffmpeg -y -hwaccel cuda -i {audio_to_convert} -acodec pcm_s16le -ac 1 -ar 16000 {output_full_path}wav"
            os.system(command)

        os.system(f'mv {os.path.join(partial_path, "clips")} {os.path.join(partial_path, "old_clips")}')
        os.system(f'mv {os.path.join(partial_path, "wav_clips")} {os.path.join(partial_path, "clips")}')
