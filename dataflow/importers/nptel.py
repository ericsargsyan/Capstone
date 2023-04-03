import os
import csv
import numpy as np
from glob import glob
import soundfile as sf
from tqdm import tqdm
from dataflow.utils import format_audio

class NptelImporter:
    def __init__(self, config):
        self.source_path = config['datasets']['Nptel']['source_path']
        self.target_dir = os.path.join(config['target_dir'], 'Nptel')
        self.csv_path = os.path.join(config['target_dir'], 'labels')
        self.samplerate = config['samplerate']
        self.duration = config['duration']

        os.makedirs(self.target_dir, exist_ok=True)
        os.makedirs(self.csv_path, exist_ok=True)

        for split in ['Test', 'Dev', 'Train']:
            if not os.path.exists(os.path.join(self.csv_path, f'{split}_nptel.csv')):
                with open(os.path.join(self.csv_path, f'{split}_nptel.csv'), 'w') as file:
                    header = csv.writer(file)
                    header.writerow(['index', 'path', 'label'])


    def import_dataset(self):
        arr = np.array([])
        for audio in glob(os.path.join(self.source_path, '*','*.wav')):
            arr = np.append(arr, audio)

        np.random.seed(5)
        train = np.random.choice(arr, int(len(arr)*0.8), replace=False)
        arr1 = np.setdiff1d(arr, train)
        test = np.random.choice(arr1, int(len(arr1)*0.6), replace=False)
        valid = np.array([i for i in arr1 if i not in test])

        self.process_data(train, 'Train')
        self.process_data(test, 'Test')
        self.process_data(valid, 'Dev')

    def process_data(self, audios, split):
        print(f"{'-' * 70}| Processing {split.upper()} Data |{'-' * 70}")
        index = 0
        os.makedirs(os.path.join(self.target_dir, split), exist_ok=True)

        for file_name in tqdm(audios):
            data = format_audio(file_name, self.samplerate, self.duration, False)
            new_audio_filepath = os.path.join(self.target_dir, split, file_name.split(os.sep)[-1])
            sf.write(new_audio_filepath, data, self.samplerate)
            labels_path = os.path.join(self.csv_path, f'{split}_nptel.csv')

            with open(labels_path, 'a+') as file:
                csvwriter = csv.writer(file)
                csvwriter.writerow([index, new_audio_filepath, 'indian'])

            index += 1