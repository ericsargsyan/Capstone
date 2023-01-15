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
        test_dataset = pd.read_table(os.path.join(self.source_path, 'en', 'test.tsv'))[['path', 'accents']]
        dev_dataset = pd.read_table(os.path.join(self.source_path, 'en', 'dev.tsv'))[['path', 'accents']]
        validated_dataset = pd.read_table(os.path.join(self.source_path, 'en', 'validated.tsv'))[['path', 'accents']]
        test_dataset = test_dataset[test_dataset['accents'].notnull()]
        dev_dataset = dev_dataset[dev_dataset['accents'].notnull()]
        validated_dataset = validated_dataset[validated_dataset['accents'].notnull()]

        # training_data = set(validated_dataset.path.to_numpy()) - set(test_dataset.path.unique()) - set(dev_dataset.path.unique())
        # train_dataset = pd.DataFrame(training_data, columns=['path', 'accents'])

        print(set(validated_dataset))

        self.process_data(test_dataset, 'TEST')
        self.process_data(dev_dataset, 'DEV')
        # self.process_data(train_dataset, 'TRAIN')

    def process_data(self, audios, split):
        print(f'------------------------------------------------------------------------'
              f'Processing {split.upper()} Data'
              f'------------------------------------------------------------------------')
        os.makedirs(os.path.join(self.target_dir, split), exist_ok=True)
        index = 0

        for file_name in tqdm(audios['path']):
            current_path = os.path.join(self.source_path, 'en', 'clips', f'{file_name[:-3]}wav')
            data = format_audio(current_path, self.samplerate, self.duration)
            accent = audios.loc[audios['path'] == file_name]['accents'][index]

            new_audio_filepath = os.path.join(self.target_dir, split, f'{file_name[:-3]}wav')
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
