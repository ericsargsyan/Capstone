import pytorch_lightning as pl
import torch
from torch import nn


class AudioModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.Net = nn.Sequential(
            nn.Linear(90000, 65536),
            nn.ReLU(),
            nn.Linear(65536, 16384),
            nn.ReLU(),
            nn.Linear(16384, 4096),
            nn.ReLU(),
            nn.Linear(4096, 576),
            nn.ReLU(),
            nn.Linear(576, 7),
            nn.ReLU(),
            nn.Softmax()
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_accuracy = 

    def forward(self, x):
        x = self.Net()
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        predict_probs = self(x)
        prediction = torch.argmax(predict_probs)
        loss = self.loss_fn(prediction, y)

        batch_accuracy =





# class LanguageDetection(pl.LightningModule, AudioModel):
#     def __init__(self):
#         super.__init__()
#         pass
#
#
# class AccentDetection(AudioModel):
#     def __init__(self):
#         pass
