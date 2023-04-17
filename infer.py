import os
from dataset import AudioDataset
from model import AudioModel
import argparse
from dataflow.utils import read_yaml
import soundfile as sf
import librosa
import torch
from glob import glob
from dataflow.utils import format_audio


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataflow_config_path',
                        type=str,
                        required=True)
    parser.add_argument('--model_config_path',
                        type=str,
                        required=True)
    parser.add_argument('--audios_dir',
                        type=str,
                        # nargs='+',
                        required=True)
    return parser.parse_args()


def detect_spoken_language_or_accent(audio, check_path, encodings):
    audio_model = AudioModel.load_from_checkpoint(check_path)
    audio_model.eval()
    y_prob = audio_model(audio)

    label = list(filter(lambda x: encodings[x] == torch.argmax(y_prob, dim=1), encodings))[0]

    return label


if __name__ == "__main__":
    parser = arg_parser()
    config = read_yaml(parser.dataflow_config_path)
    model_config = read_yaml(parser.model_config_path)
    audio_dir = parser.audios_dir

    audios = []

    for audio_path in glob(f'{audio_dir}{os.sep}*.wav'):
        data = format_audio(audio_path,
                            self_samplerate=config['samplerate'],
                            resample=True,
                            self_duration=config['duration'])
        data = torch.tensor(data)
        audios.append(data)

        # data_raw, samplerate = sf.read(audio_path)
        # data = librosa.resample(data_raw, target_sr=config['sr'], orig_sr=samplerate)
        # duration = data.shape[0] / config['sr']
        # data = torch.tensor(data[:config['sr'] * 2], dtype=torch.float32).view(1, -1)

    data = torch.cat(audios)

    print(data.shape)

    checkpoint_path = '/home/capstone/Desktop/Krisp/Capstone/language_logs/version_3/checkpoints/epoch=02-val_epoch_accuracy=0.000000.ckpt'

    # model = AudioModel.load_from_checkpoint(checkpoint_path, read_yaml('./configs/model/net.yaml'),
    #                    model_config['audio_processor'],
    #                    model_config['sr'],
    #                    max(model_config['encodings']['language_detection'].values()) + 1,
    #                    model_config['learning_rate'])
    # model.eval()

    # y_prob = model(data)
    # print(y_prob)
    # #
    # for idx, y in enumerate(y_prob):
    #     if y >= 0.5:
    #         print(f'Label of {audio_paths[idx].split(os.sep)[-1]} - M')
    #     else:
    #         print(f'Label of {audio_paths[idx].split(os.sep)[-1]} - F')
