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

        if config['convert']['wav']:
            self.file_ending = 'wav'
            self.resample = False
        else:
            self.file_ending = 'mp3'
            self.resample = True

        for split in ['Test', 'Dev', 'Train']:
            if not os.path.exists(os.path.join(self.csv_path, f'{split}_mozilla.csv')):
                with open(os.path.join(self.csv_path, f'{split}_mozilla.csv'), 'w') as file:
                    header = csv.writer(file)
                    header.writerow(['index', 'path', task.split('_')[0]])

    def import_datasets(self, languages):
        for language in languages:
            print(f"{'-' * 70}| Processing {language.upper()} Data |{'-' * 70}")
            tsv_path = os.path.join(self.source_path, language)

            self.test_dataset = pd.read_table(os.path.join(tsv_path, 'test.tsv'))[['path', 'accents']]
            self.dev_dataset = pd.read_table(os.path.join(tsv_path, 'dev.tsv'))[['path', 'accents']]
            validated_dataset = pd.read_table(os.path.join(tsv_path, 'validated.tsv'))[['path', 'accents']]

            self.train_dataset = validated_dataset[~validated_dataset.path.isin(self.test_dataset.path)]
            self.train_dataset = self.train_dataset[~self.train_dataset.path.isin(self.dev_dataset.path)]

            self._task_related_processing()

            self.process_data(self.train_dataset, 'Train', language)
            self.process_data(self.test_dataset, 'Test', language)
            self.process_data(self.dev_dataset, 'Dev', language)

    def _get_audio_filepath_label(self, audios, split, file_name, index, language):
        raise NotImplementedError

    def _task_related_processing(self):
        raise NotImplementedError

    def process_data(self, audios, split, language):
        print(f"{'-' * 70}| Processing {split.upper()} Data |{'-' * 70}")
        index = 0

        for file_name in tqdm(audios['path']):
            current_path = os.path.join(self.source_path, language, 'clips', f'{file_name[:-3]}{self.file_ending}')
            data = format_audio(current_path, self.samplerate, self.duration, self.resample)

            new_audio_filepath, label = self._get_audio_filepath_label(audios, split, file_name, index, language)
            sf.write(new_audio_filepath, data, self.samplerate)
            labels_path = os.path.join(self.csv_path, f'{split}_mozilla.csv')

            with open(labels_path, 'a+') as file:
                csvwriter = csv.writer(file)
                csvwriter.writerow([index, new_audio_filepath, label])

            index += 1


class LanguageImporter(MozillaCVImporter):
    def __init__(self, config):
        super().__init__(config, 'language_detection')
        self.languages = config['languages']

    def import_dataset(self):
        super(LanguageImporter, self).import_datasets(self.languages)

    def _task_related_processing(self):
        pass

    def _get_audio_filepath_label(self, audios, split, file_name, index, language):
        new_audio_filepath = os.path.join(self.target_dir, language, split, f'{file_name[:-3]}wav')
        return new_audio_filepath, language

    def process_data(self, audios, split, language):
        os.makedirs(os.path.join(self.target_dir, language, split), exist_ok=True)
        super(LanguageImporter, self).process_data(audios, split, language)


class AccentImporter(MozillaCVImporter):
    def __init__(self, config):
        super().__init__(config, 'accent_detection')
        # self.accents = config['accents']
        self.languages = ['en']

    def _task_related_processing(self):
        self.test_dataset = self.test_dataset[self.test_dataset['accents'].notnull()].reset_index()
        self.train_dataset = self.train_dataset[self.train_dataset['accents'].notnull()].reset_index()
        self.dev_dataset = self.dev_dataset[self.dev_dataset['accents'].notnull()].reset_index()

    def _get_audio_filepath_label(self, audios, split, file_name, index, language):
        accent = audios.loc[audios['path'] == file_name]['accents'][index]
        new_audio_filepath = os.path.join(self.target_dir, split, f'{file_name[:-3]}wav')

        return new_audio_filepath, accent

    def import_dataset(self):
        super(AccentImporter, self).import_datasets(self.languages)

    def process_data(self, audios, split, language):
        os.makedirs(os.path.join(self.target_dir, split), exist_ok=True)
        super(AccentImporter, self).process_data(audios, split, language)
