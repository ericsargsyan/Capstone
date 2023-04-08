import os
import csv
import numpy as np
from glob import glob
import soundfile as sf
from tqdm import tqdm
from dataflow.utils import format_audio, data_spliter


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
        print(f"{'-' * 70}| Processing NPTEL Data |{'-' * 70}")
        arr = []
        for audio in tqdm(glob(os.path.join(self.source_path, 'nptel-test', 'wav', '*.wav'))):
            arr.append(audio)
        arr = np.array(arr)
        np.random.seed(12)
        train, test = data_spliter(arr, 0.8)
        test, valid = data_spliter(test, 0.6)

        self.process_data(train, 'Train')
        self.process_data(test, 'Test')
        self.process_data(valid, 'Dev')

    def process_data(self, audios, split):
        print(f"{'-' * 70}| Processing {split.upper()} Data |{'-' * 70}")
        index = 0
        crashed_files = 0
        os.makedirs(os.path.join(self.target_dir, split), exist_ok=True)

        for file_name in tqdm(audios):
            try:
                data = format_audio(file_name, self.samplerate, self.duration, False)
            except sf.LibsndfileError:
                crashed_files += 1
                continue
            new_audio_filepath = os.path.join(self.target_dir, split, file_name.split(os.sep)[-1])
            sf.write(new_audio_filepath, data, self.samplerate)
            labels_path = os.path.join(self.csv_path, f'{split}_nptel.csv')

            with open(labels_path, 'a+') as file:
                csvwriter = csv.writer(file)
                csvwriter.writerow([index, new_audio_filepath, 'indian'])

            index += 1
        print(f"There are {crashed_files} crashed files in {split}")