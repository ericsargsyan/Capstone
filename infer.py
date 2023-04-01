import os
from dataset import LanguageDataset, AccentDataset
from model import AudioModel
import argparse
from dataflow.utils import read_yaml
import soundfile as sf
import librosa
import torch
from glob import glob


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        type=str,
                        required=True)
    parser.add_argument('--audios_dir',
                        type=str,
                        nargs='+',
                        required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parser = arg_parser()
    config = read_yaml(parser.config_path)
    audio_dir = parser.audios_dir

    audios = []

    for audio_path in glob(audio_dir):
        data_raw, samplerate = sf.read(audio_path)
        data = librosa.resample(data_raw, target_sr=config['sr'], orig_sr=samplerate)
        duration = data.shape[0] / config['sr']
        data = torch.tensor(data[:config['sr'] * 2], dtype=torch.float32).view(1, -1)
        audios.append(data)

    data = torch.cat(audios)

    check_path = ''

    model = AudioModel.load_from_checkpoint(check_path)
    model.eval()

    y_prob = model(data)
    print(y_prob)
