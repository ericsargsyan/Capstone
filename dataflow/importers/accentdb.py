import os
import csv
import numpy as np
from glob import glob
import soundfile as sf
from tqdm import tqdm
from dataflow.utils import format_audio, data_spliter


class AccentDBImporter:
    def __init__(self, config):
        self.source_path = config['datasets']['AccentDB']['source_path']
        self.target_dir = os.path.join(config['target_dir'], 'AccentDB')
        self.csv_path = os.path.join(config['target_dir'], 'labels')
        self.samplerate = config['samplerate']
        self.duration = config['duration']

        os.makedirs(self.target_dir, exist_ok=True)
        os.makedirs(self.csv_path, exist_ok=True)

        for split in ['Test', 'Dev', 'Train']:
            if not os.path.exists(os.path.join(self.csv_path, f'{split}_accentdb.csv')):
                with open(os.path.join(self.csv_path, f'{split}_accentdb.csv'), 'w') as file:
                    header = csv.writer(file)
                    header.writerow(['index', 'path', 'label'])

    def import_dataset(self):
        print(f"{'-' * 70}| Processing AccentDB Data |{'-' * 70}")
        train_audios, val_audios, test_audios = [], [], []
        accents = [accent.split('/')[-1] for accent in glob(os.path.join(self.source_path, '*', '*'))]
        for accent_to_proceed in accents:
            audios = glob(os.path.join(self.source_path, 'data', accent_to_proceed, '*', '*'))
            np.random.seed(12)
            train_audios, test_audios = data_spliter(audios, 0.8)
            test_audios, val_audios = data_spliter(test_audios, 0.6)

        self.process_data(train_audios, 'Train')
        self.process_data(test_audios, 'Test')
        self.process_data(val_audios, 'Dev')

    def process_data(self, audios, split):
        crashed_files = 0
        print(f"{'-' * 70}| Processing {split.upper()} Data |{'-' * 70}")
        index = 0
        os.makedirs(os.path.join(self.target_dir, split), exist_ok=True)

        for file_name in tqdm(audios):
            data = format_audio(file_name, self.samplerate, self.duration, False)
            new_audio_filepath = os.path.join(self.target_dir, split, file_name.split(os.sep)[-1])

            try:
                sf.write(new_audio_filepath, data, self.samplerate)
            except sf.LibsndfileError:
                crashed_files += 1
                continue

            labels_path = os.path.join(self.csv_path, f'{split}_accentdb.csv')

            with open(labels_path, 'a+') as file:
                csvwriter = csv.writer(file)
                csvwriter.writerow([index, new_audio_filepath, new_audio_filepath.split('/')[-1].split('_')[0]])

            index += 1
        print(f"There are {crashed_files} crashed files in {split}")
