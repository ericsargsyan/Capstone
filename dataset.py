from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torch


class AccentDataset(Dataset):
    def __init__(self, paths, label):
        self.data = pd.concat((pd.read_csv(data) for data in paths), ignore_index=True)
        self.labels = pd.DataFrame.from_dict(label, orient='index', columns=['label'])
        self.data = pd.merge(self.data, self.labels, left_on='accent', right_index=True)

    def __getitem__(self, idx):
        waveform, _ = torchaudio.load(self.data['path'][idx])
        x = waveform.view(-1)
        y = torch.tensor(self.data['label'], dtype=torch.long)

    def __len__(self):
        return len(self.data)


class LanguageDataset(Dataset):
    def __init__(self, path, label):
        self.data = pd.read_csv(path)
        self.labels = pd.DataFrame.from_dict(label, orient='index', columns=['label'])
        self.data = pd.merge(self.data, self.labels, left_on='language', right_index=True)

    def __getitem__(self, idx):
        waveform, _ = torchaudio.load(self.data['path'][idx])
        x = waveform.view(-1)
        y = torch.tensor(self.data['label'], dtype=torch.long)

        return x, y[idx]

    def __len__(self):
        return len(self.data)
