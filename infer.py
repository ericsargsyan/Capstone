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
    # parser.add_argument('--check_path',
    #                     type=str,
    #                     required=True)
    parser.add_argument('--audios_dir',
                        type=str,
                        # nargs='+',
                        required=True)
    return parser.parse_args()


def detect_spoken_language_or_accent(audio, check_path, encodings):
    audio_model = AudioModel.load_from_checkpoint(checkpoint_path=check_path,
                                                  model_config=read_yaml('./configs/model/net.yaml'),
                                                  processor_config=model_config['audio_processor'],
                                                  sr=model_config['sr'],
                                                  number_of_labels=model_config['encodings'][model_config['task']],
                                                  learning_rate=model_config['learning_rate'])
    audio_model.eval()
    y_prob = audio_model(audio)

    label = list(filter(lambda x: encodings[x] == torch.argmax(y_prob, dim=1), encodings))[0]

    return label


if __name__ == "__main__":
    parser = arg_parser()
    dataflow_config = read_yaml(parser.dataflow_config_path)
    model_config = read_yaml(parser.model_config_path)
    audio_dir = parser.audios_dir

    audios = []

    for audio_path in glob(f'{audio_dir}{os.sep}*.mp3'):
        data = format_audio(audio_path,
                            self_samplerate=dataflow_config['samplerate'],
                            resample=True,
                            self_duration=dataflow_config['duration'])
        data = torch.tensor(data, dtype=torch.float32).view(1, -1)
        audios.append(data)

    data = torch.cat(audios, dim=0)
    print(len(audios), audios[0].shape)
    print(data.shape)

    checkpoint_path = '/Users/eric/Desktop/Krisp/Capstone/language_logs/version_3/checkpoints/epoch=01-val_epoch_accuracy=0.930701.ckpt'

    number_of_labels = max(model_config['encodings'][model_config['task']].values()) + 1

    model = AudioModel.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                            model_config=read_yaml('./configs/model/net.yaml'),
                                            processor_config=model_config['audio_processor'],
                                            sr=model_config['sr'],
                                            number_of_labels=number_of_labels,
                                            learning_rate=model_config['learning_rate'])
    model.eval()
    y_prob = model(data)

    print(data.shape, y_prob)

    # print(y_prob)

    # for idx, y in enumerate(y_prob):
    #     if y >= 0.5:
    #         print(f'Label of {audio_paths[idx].split(os.sep)[-1]} - M')
    #     else:
    #         print(f'Label of {audio_paths[idx].split(os.sep)[-1]} - F')


