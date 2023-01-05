from torch.utils.data import Dataset
import pandas as pd


class AccentDataset(Dataset):
    def __init__(self, paths):
        self.data = pd.concat((pd.read_csv(data) for data in paths), ignore_index=True)

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return len(self.data)


class LanguageDataset(Dataset):
    def __init__(self, paths):
        self.data = pd.concat((pd.read_csv(data) for data in paths), ignore_index=True)

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return len(self.data)
