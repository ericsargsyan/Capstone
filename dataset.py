from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torch
import os
from sklearn.preprocessing import LabelEncoder


class AccentDataset(Dataset):
    def __init__(self, paths):
        self.data = pd.concat((pd.read_csv(data) for data in paths), ignore_index=True)


    def __getitem__(self, idx):
        pass

    def __len__(self):
        return len(self.data)


class LanguageDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.data['language'] = self.data['language'].astype('category')
        self.data['encoded_language'] = self.data['language'].cat.codes
        dummy_data = self.data[['language', 'encoded_language']].drop_duplicates().reset_index(drop=True)
        dummy_data.to_csv(os.path.join(os.getcwd(), 'encoded_labels.csv'))

    def __getitem__(self, idx):
        waveform, _ = torchaudio.load(self.data['path'][idx])
        x = waveform.view(-1)
        y = torch.tensor(self.data['encoded_language'], dtype=torch.long)

        return x, y[idx]#.unsqueeze(-1)

    def __len__(self):
        return len(self.data)
