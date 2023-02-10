import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy


class AudioModel(pl.LightningModule):
    def __init__(self, number_of_labels, learning_rate):
        super().__init__()
        self.Net = nn.Sequential(
            nn.Linear(80000, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, number_of_labels)
            # nn.Softmax()
        )
        self.learning_rate = learning_rate

        self.loss_fn = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task='multiclass', num_classes=number_of_labels)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=number_of_labels)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=number_of_labels)

    def forward(self, x):
        x = self.Net(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        predict = self(x)
        prediction = torch.argmax(predict, dim=1)

        loss = self.loss_fn(predict, y)
        batch_accuracy = self.train_accuracy(prediction, y)

        self.log('train_step_accuracy', batch_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_step_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def training_epoch_end(self, outputs):
        train_epoch_acc = self.train_accuracy.compute()
        self.train_accuracy.reset()

        self.log('train_epoch_accuracy', train_epoch_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        predict = self(x)
        prediction = torch.argmax(predict, dim=1)

        loss = self.loss_fn(predict, y)
        print(predict.shape)
        print(prediction.shape, y.shape)
        print(prediction)
        print(y)
        batch_accuracy = self.val_accuracy(prediction, y)

        self.log('val_step_accuracy', batch_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_step_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def validation_epoch_end(self, outputs):
        val_epoch_acc = self.val_accuracy.compute()
        self.val_accuracy.reset()

        self.log('val_epoch_accuracy', val_epoch_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, test_batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)


# class LanguageDetection(pl.LightningModule, AudioModel):
#     def __init__(self):
#         super.__init__()
#         pass
#
#
# class AccentDetection(AudioModel):
#     def __init__(self):
#         pass
