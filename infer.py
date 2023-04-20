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
    parser.add_argument('--check_path',
                        type=str,
                        required=True)
    parser.add_argument('--audios_dir',
                        type=str,
                        required=True)
    return parser.parse_args()


def detect_spoken_language_or_accent(data, encodings, check_path, model_config, processor_config, sr, labels, lr):
    audio_model = AudioModel.load_from_checkpoint(checkpoint_path=check_path,
                                                  model_config=read_yaml(model_config),
                                                  processor_config=processor_config,
                                                  sr=sr,
                                                  number_of_labels=labels,
                                                  learning_rate=lr)
    audio_model.eval()
    y_prob = audio_model(data)

    label = list(filter(lambda x: encodings[x] == torch.argmax(y_prob, dim=1), encodings))[0]

    return label


if __name__ == "__main__":
    parser = arg_parser()
    dataflow_config = read_yaml(parser.dataflow_config_path)
    model_config = read_yaml(parser.model_config_path)
    audio_dir = parser.audios_dir

    task = model_config['task']
    number_of_labels = max(model_config['encodings'][task].values()) + 1
    encodings = model_config['encodings'][task]

    audios = []
    audio_names = []

    for audio_path in glob(f'{audio_dir}{os.sep}*.mp3'):
        data = format_audio(audio_path,
                            self_samplerate=dataflow_config['samplerate'],
                            resample=True,
                            self_duration=dataflow_config['duration'])
        data = torch.tensor(data, dtype=torch.float32).view(1, -1)
        audios.append(data)
        audio_names.append(os.path.basename(audio_path)[:-4])

    data = torch.cat(audios, dim=0)

    check_abs_path = os.path.join(model_config['log_dir'][task], parser.check_path, 'checkpoints')
    checkpoint_files = glob(os.path.join(check_abs_path, '*.ckpt'))

    accuracies = [float(checkpoint.split('=')[2][:-5]) for checkpoint in checkpoint_files]
    checkpoint_path = checkpoint_files[accuracies.index(max(accuracies))]

    model = AudioModel.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                            model_config=read_yaml('./configs/model/net.yaml'),
                                            processor_config=model_config['audio_processor'],
                                            sr=model_config['sr'],
                                            number_of_labels=number_of_labels,
                                            learning_rate=model_config['learning_rate'])
    model.eval()
    y_prob = model(data)

    speech = task.split('_')[0]

    for index, y in enumerate(torch.argmax(y_prob, dim=1)):
        label = list(filter(lambda x: encodings[x] == y, encodings))[0]
        print(f'Spoken {speech} of {audio_names[index]}: {label}')
