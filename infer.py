import os
import torch
import argparse
from glob import glob
from model import AudioModel
from dataflow.utils import read_yaml
from dataflow.utils import format_audio
from utils import get_best_checkpoint


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataflow_config_path',
                        type=str,
                        required=True)
    # parser.add_argument('--model_config_path',
    #                     type=str,
    #                     required=True)
    parser.add_argument('--check_path',
                        type=str,
                        required=True)
    # parser.add_argument('--audios_dir',
    #                     type=str,
    #                     required=True)

    return parser.parse_args()


def detect_spoken_language_or_accent(data, model, encodings):
    data = torch.tensor(data, dtype=torch.float32).view(1, -1)
    y_prob = model(data)
    label = list(filter(lambda x: encodings[x] == torch.argmax(y_prob, dim=1).squeeze(), encodings))[0]

    return label


if __name__ == "__main__":
    parser = arg_parser()
    dataflow_config = read_yaml(parser.dataflow_config_path)
    model_config = read_yaml(dataflow_config['model_config_path'])
    audio_dir = dataflow_config['inference_audio_dir_path']

    task = model_config['task']
    number_of_labels = max(model_config['encodings'][task].values()) + 1
    encodings = model_config['encodings'][task]

    audios = []
    audio_names = []

    for audio_path in glob(f'{audio_dir}{os.sep}*'):
        data = format_audio(audio_path,
                            self_samplerate=dataflow_config['samplerate'],
                            resample=True,
                            self_duration=dataflow_config['duration'])
        data = torch.tensor(data, dtype=torch.float32).view(1, -1)

        audios.append(data)
        audio_names.append(os.path.basename(audio_path)[:-4])

    data = torch.cat(audios, dim=0)

    checkpoint_path = get_best_checkpoint(model_config['log_dir'][task], parser.check_path)

    model = AudioModel.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                            model_config=read_yaml('./configs/model/net.yaml'),
                                            processor_config=model_config['audio_processor'],
                                            sr=model_config['sr'],
                                            number_of_labels=number_of_labels,
                                            learning_rate=model_config['learning_rate'],
                                            encodings=model_config['encodings'][task])
    model.eval()
    y_prob = model(data)

    speech = task.split('_')[0]

    for index, y in enumerate(torch.argmax(y_prob, dim=1)):
        label = list(filter(lambda x: encodings[x] == y, encodings))[0]
        print(f'Spoken {speech} of {audio_names[index]}: {label}')
