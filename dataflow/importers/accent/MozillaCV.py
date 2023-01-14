import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
from dataflow.utils import format_audio


class MozillaCVAImporter:
    def __init__(self, config):
        self.source_path = config['datasets']['MozillaCV']['source_path']
        self.target_dir = os.path.join(config['target_dir'], 'accent_detection', 'MozillaCV')
        self.csv_path = os.path.join(config['target_dir'], 'accent_detection', 'labels')
        os.makedirs(self.target_dir, exist_ok=True)
        os.makedirs(self.csv_path, exist_ok=True)
        self.samplerate = config['samplerate']
        self.duration = config['duration']

    def import_dataset(self):
        dataset = pd.read_table(os.path.join(self.source_path, 'en', 'validated.tsv'))[['path', 'accents']]
        dataset = dataset[dataset['accents'].notnull()]
        dataset = dataset.reset_index()
        train_val, test = np.split(dataset.sample(frac=1, random_state=12, replace=False), [int(len(dataset) * 0.8)])
        train, val = np.split(train_val.sample(frac=1, random_state=12, replace=False), [int(len(train_val) * 0.9)])
        train, val, test = train.reset_index(), val.reset_index(), test.reset_index()

        self.process_data(train, 'TRAIN')
        self.process_data(val, 'VAL')
        self.process_data(test, 'TEST')

    def process_data(self, audios, split):
        os.makedirs(os.path.join(self.target_dir, split), exist_ok=True)
        index = 0

        for file_name in tqdm(audios['path']):
            current_path = os.path.join(self.source_path, 'en', 'clips', file_name)
            data = format_audio(current_path, self.samplerate, self.duration)
            accent = audios.loc[audios['path'] == file_name]['accents'][index]

            new_audio_filepath = os.path.join(self.target_dir, split, f'{file_name}.wav')
            sf.write(new_audio_filepath, data, self.samplerate)

            labels_path = os.path.join(self.csv_path, f'{split}_mozilla.csv')

            if not os.path.exists(labels_path):
                with open(labels_path, 'w') as file:
                    header = csv.writer(file)
                    header.writerow(['index', 'path', 'accent'])

            with open(labels_path, 'a+') as file:
                csvwriter = csv.writer(file)
                csvwriter.writerow([index, new_audio_filepath, accent])

            index += 1
