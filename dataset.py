from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torch


class AudioDataset(Dataset):
    def __init__(self, paths, label, task):
        self.data = pd.concat((pd.read_csv(data) for data in paths), ignore_index=True)
        # self.labels = pd.DataFrame.from_dict(label, orient='index', columns=['label'])
        # self.data = pd.merge(self.data, self.labels, left_on=task.split('_')[0], right_index=True)
        column_name = task.split('_')[0]
        self.data['label'] = self.data[column_name].apply(lambda x: label[x] if x in label else max(label.values()))
        self.y = torch.tensor(self.data['label'], dtype=torch.long)

    def __getitem__(self, idx):
        waveform, _ = torchaudio.load(self.data['path'][idx])
        x = waveform.view(-1)

        return x, self.y[idx]

    def __len__(self):
        return len(self.data)
