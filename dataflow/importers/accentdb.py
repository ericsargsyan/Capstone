import os
import csv
import numpy as np
from glob import glob
import soundfile as sf
from tqdm import tqdm
from dataflow.utils import format_audio


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

        train_audios = np.array([])
        val_audios = np.array([])
        test_audios = np.array([])

        accents = [accent.split('/')[-1] for accent in glob(os.path.join(self.source_path, '*'))]

        for accent_to_proceed in accents:
            audios = glob(os.path.join(self.source_path, accent_to_proceed, '*', '*'))

            np.random.seed(1220)

            train_audios = np.append(train_audios, np.random.choice(audios, int(len(audios) * 0.8), replace=False))
            diff = np.setdiff1d(audios, train_audios)
            test_audios = np.append(test_audios, np.random.choice(diff, int(len(diff) * 0.5), replace=False))
            val_audios = np.array([i for i in diff if i not in test_audios])

        self.process_data(train_audios, 'Train')
        self.process_data(test_audios, 'Test')
        self.process_data(val_audios, 'Val')

    def process_data(self, audios, split):
        print(f"{'-' * 70}| Processing {split.upper()} Data |{'-' * 70}")
        index = 0
        os.makedirs(os.path.join(self.target_dir, split), exist_ok=True)

        for file_name in tqdm(audios):
            data = format_audio(file_name, self.samplerate, self.duration, False)
            new_audio_filepath = os.path.join(self.target_dir, split, file_name.split(os.sep)[-1])
            sf.write(new_audio_filepath, np.ravel(data), self.samplerate)
            labels_path = os.path.join(self.csv_path, f'{split}_accentdb.csv')

            with open(labels_path, 'a+') as file:
                csvwriter = csv.writer(file)
                csvwriter.writerow([index, new_audio_filepath, new_audio_filepath.split('/')[-1].split('_')[0]])

            index += 1
