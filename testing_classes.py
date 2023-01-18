import os
import csv
import soundfile as sf
import numpy as np
from tqdm import tqdm
import pandas as pd
from dataflow.utils import format_audio


class MozillaCVLImporter:
    def __init__(self, config):
        self.source_path = config['datasets']['MozillaCV']['source_path']
        self.target_dir = os.path.join(config['target_dir'], 'language_detection', 'MozillaCV')
        self.csv_path = os.path.join(config['target_dir'], 'language_detection', 'labels')
        os.makedirs(self.target_dir, exist_ok=True)
        os.makedirs(self.csv_path, exist_ok=True)
        self.samplerate = config['samplerate']
        self.duration = config['duration']
        self.languages = config['languages']

        if config['system'] == 'Linux':
            self.file_ending = 'wav'
            self.resample = False
        else:
            self.file_ending = 'mp3'
            self.resample = True

        for split in ['TEST', 'DEV', 'TRAIN']:
            if not os.path.exists(os.path.join(self.csv_path, f'{split}_mozilla.csv')):
                with open(os.path.join(self.csv_path, f'{split}_mozilla.csv'), 'w') as file:
                    header = csv.writer(file)
                    header.writerow(['index', 'path', 'language'])

    def import_dataset(self):
        for language in self.languages:
            print(f'------------------------------------------------------------------------'
                  f'    Processing {language.upper()} Data    '
                  f'------------------------------------------------------------------------')
            test_dataset = pd.read_table(os.path.join(self.source_path, language, 'test.tsv'))[['path']]
            dev_dataset = pd.read_table(os.path.join(self.source_path, language, 'dev.tsv'))[['path']]
            validated_dataset = pd.read_table(os.path.join(self.source_path, language, 'validated.tsv'))[['path']]

            train_dataset = validated_dataset[~validated_dataset.path.isin(test_dataset.path)]
            train_dataset = train_dataset[~train_dataset.path.isin(dev_dataset.path)]

            self.process_data(test_dataset, language, 'TEST')
            self.process_data(dev_dataset, language, 'DEV')
            self.process_data(train_dataset, language, 'TRAIN')

    def process_data(self, audios, language, split):
        print(f'------------------------------------------------------------------------'
              f'Processing {split.upper()} Data'
              f'------------------------------------------------------------------------')
        os.makedirs(os.path.join(self.target_dir, language, split), exist_ok=True)
        index = 0

        for file_name in tqdm(audios['path']):
            current_path = os.path.join(self.source_path, language, 'clips', f'{file_name[:-3]}{self.file_ending}')
            data = format_audio(current_path, self.samplerate, self.duration, resample=self.resample)

            new_audio_filepath = os.path.join(self.target_dir, language, split, f'{file_name[:-3]}{self.file_ending}')
            sf.write(new_audio_filepath, data, self.samplerate)

            labels_path = os.path.join(self.csv_path, f'{split}_mozilla.csv')

            with open(labels_path, 'a+') as file:
                csvwriter = csv.writer(file)
                csvwriter.writerow([index, new_audio_filepath, language])

            index += 1
