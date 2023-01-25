import os
import csv
import soundfile as sf
from tqdm import tqdm
import pandas as pd
from dataflow.utils import format_audio


class MozillaCVImporter:
    def __init__(self, config, task):
        self.source_path = config['datasets']['MozillaCV']['source_path']
        self.target_dir = os.path.join(config['target_dir'], task, 'MozillaCV')
        self.csv_path = os.path.join(config['target_dir'], task, 'labels')
        os.makedirs(self.target_dir, exist_ok=True)
        os.makedirs(self.csv_path, exist_ok=True)
        self.samplerate = config['samplerate']
        self.duration = config['duration']

        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None

        self.task = task

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
                    header.writerow(['index', 'path', task.split('_')[0]])

    def import_datasets(self, languages):
        for language in languages:
            print(f'------------------------------------------------------------------------'
                  f'    Processing {language.upper()} Data    '
                  f'------------------------------------------------------------------------')
            self.test_dataset = pd.read_table(os.path.join(self.source_path, language, 'test.tsv'))[['path', 'accents']]
            self.dev_dataset = pd.read_table(os.path.join(self.source_path, language, 'dev.tsv'))[['path', 'accents']]
            validated_dataset = pd.read_table(os.path.join(self.source_path, language, 'validated.tsv'))[['path', 'accents']]

            self.train_dataset = validated_dataset[~validated_dataset.path.isin(self.test_dataset.path)]
            self.train_dataset = self.train_dataset[~self.train_dataset.path.isin(self.dev_dataset.path)]

            if self.task == 'language_detection':
                self.process_data(self.test_dataset, 'TEST', language)
                self.process_data(self.dev_dataset, 'DEV', language)
                self.process_data(self.train_dataset, 'TRAIN', language)

    def process_data(self, audios, split, language):
        print(f'------------------------------------------------------------------------'
              f' Processing {split.upper()} Data '
              f'------------------------------------------------------------------------')
        os.makedirs(os.path.join(self.target_dir, language, split), exist_ok=True)

        index = 0

        if self.task == 'language_detection':
            for file_name in tqdm(audios['path'][:1500]):
                current_path = os.path.join(self.source_path, language, 'clips', f'{file_name[:-3]}{self.file_ending}')
                data = format_audio(current_path, self.samplerate, self.duration, self.resample)

                new_audio_filepath = os.path.join(self.target_dir, language, split, f'{file_name[:-3]}{self.file_ending}')
                sf.write(new_audio_filepath, data, self.samplerate)

                labels_path = os.path.join(self.csv_path, f'{split}_mozilla.csv')

                with open(labels_path, 'a+') as file:
                    csvwriter = csv.writer(file)
                    csvwriter.writerow([index, new_audio_filepath, language])

                index += 1


class LanguageImporter(MozillaCVImporter):
    def __init__(self, config):
        super().__init__(config, 'language_detection')
        self.languages = config['languages']

    def import_dataset(self):
        super(LanguageImporter, self).import_datasets(self.languages)

    # def process_data(self, audios, language, split):
    #     super(LanguageImporter, self).import_datasets(self.languages)


class AccentImporter(MozillaCVImporter):
    def __init__(self, config):
        super().__init__(config, 'accent_detection')
        self.accents = config['accents']
        self.languages = ['en']

    def import_dataset(self):
        super(AccentImporter, self).import_datasets(self.languages)

        self.test_dataset = self.test_dataset[self.test_dataset['accents'].notnull()]
        self.train_dataset = self.train_dataset[self.train_dataset['accents'].notnull()]
        self.dev_dataset = self.dev_dataset[self.dev_dataset['accents'].notnull()]

        self.train_dataset = self.train_dataset.reset_index()
        self.test_dataset = self.test_dataset.reset_index()
        self.dev_dataset = self.dev_dataset.reset_index()

        self.process_data(self.test_dataset, split='TEST')
        self.process_data(self.dev_dataset, split='DEV')
        self.process_data(self.train_dataset, split='TRAIN')

    def process_data(self, audios, split, language='en'):
        print(f'------------------------------------------------------------------------'
              f' Processing {split.upper()} Data '
              f'------------------------------------------------------------------------')
        os.makedirs(os.path.join(self.target_dir, split), exist_ok=True)

        index = 0

        for file_name in tqdm(audios['path'][:1500]):
            current_path = os.path.join(self.source_path, language, 'clips', f'{file_name[:-3]}{self.file_ending}')
            data = format_audio(current_path, self.samplerate, self.duration, self.resample)
            accent = audios.loc[audios['path'] == file_name]['accents'][index]

            new_audio_filepath = os.path.join(self.target_dir, split, f'{file_name[:-3]}{self.file_ending}')
            sf.write(new_audio_filepath, data, self.samplerate)

            labels_path = os.path.join(self.csv_path, f'{split}_mozilla.csv')

            with open(labels_path, 'a+') as file:
                csvwriter = csv.writer(file)
                csvwriter.writerow([index, new_audio_filepath, accent])

            index += 1
