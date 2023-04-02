from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torch


class AudioDataset(Dataset):
    def __init__(self, paths, label, task):
        print('1')
        self.data = pd.concat((pd.read_csv(data) for data in paths), ignore_index=True)
        print('2')
        self.labels = pd.DataFrame.from_dict(label, orient='index', columns=['label'])
        print('3')
        self.data = pd.merge(self.data, self.labels, left_on=task.split('_')[0], right_index=True)
        print('4')

    def __getitem__(self, idx):
        waveform, _ = torchaudio.load(self.data['path'][idx])
        x = waveform.view(-1)
        y = torch.tensor(self.data['label'], dtype=torch.long)

        return x, y[idx]

    def __len__(self):
        return len(self.data)
