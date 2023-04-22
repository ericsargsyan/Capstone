import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy
from net.speaker_net import SpeakerNet
from net.layers import NormalizedMelSpectogram
from torchmetrics.classification import MulticlassF1Score


class AudioModel(pl.LightningModule):
    def __init__(self, model_config, processor_config, sr, number_of_labels, learning_rate, encodings):
        super().__init__()
        self.net = SpeakerNet(model_config, number_of_labels)
        win_length = int(processor_config.pop('win_length') * sr)
        hop_length = int(processor_config.pop('hop_length') * sr)
        self.audio_processor = NormalizedMelSpectogram(win_length=win_length,
                                                       hop_length=hop_length,
                                                       window_fn=torch.hann_window,
                                                       **processor_config)
        self.learning_rate = learning_rate

        self.loss_fn = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task='multiclass', num_classes=number_of_labels)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=number_of_labels)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=number_of_labels)
        self.train_f1 = MulticlassF1Score(num_classes=number_of_labels, average=None)
        self.val_f1 = MulticlassF1Score(num_classes=number_of_labels, average=None)
        self.test_f1 = MulticlassF1Score(num_classes=number_of_labels, average=None)

        self.index_to_label = {}

        for key, value in encodings.items():
            if value not in self.index_to_label:
                self.index_to_label[value] = key
            else:
                self.index_to_label[value] = self.index_to_label[value]+f', {key}'

    def forward(self, x):
        x = self.audio_processor(x)
        x = self.net(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        predict = self(x)
        prediction = torch.argmax(predict, dim=1)

        loss = self.loss_fn(predict, y)
        batch_accuracy = self.train_accuracy(prediction, y)
        batch_train_f1 = self.val_f1(prediction, y)

        if (self.global_step+1) % self.trainer.log_every_n_steps == 0:
            for i, f1 in enumerate(batch_train_f1):
                self.logger.experiment.add_scalars(f'train_step_f1', {f"{self.index_to_label[i]}": f1}, self.global_step)

        self.log('train_step_accuracy', batch_accuracy, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_step_loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        return loss

    def training_epoch_end(self, outputs):
        train_epoch_acc = self.train_accuracy.compute()
        self.train_accuracy.reset()

        loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log('train_epoch_accuracy', train_epoch_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_epoch_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        predict = self(x)
        prediction = torch.argmax(predict, dim=1)

        loss = self.loss_fn(predict, y)
        batch_accuracy = self.val_accuracy(prediction, y)
        self.val_f1(prediction, y)

        self.log('val_step_accuracy', batch_accuracy, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('val_step_loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        return loss

    def validation_epoch_end(self, outputs):
        val_epoch_acc = self.val_accuracy.compute()
        self.val_accuracy.reset()
        val_epoch_f1 = self.val_f1.compute().tolist()
        self.val_f1.reset()

        loss = sum(outputs) / len(outputs)
        # loss = torch.stack([x['loss'] for x in outputs]).mean()

        for i, f1 in enumerate(val_epoch_f1):
            self.logger.experiment.add_scalars(f'val_f1_epoch', {f"{self.index_to_label[i]}": f1}, self.global_step)

        self.log('val_epoch_accuracy', val_epoch_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_epoch_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        predict = self(x)
        prediction = torch.argmax(predict, dim=1)

        loss = self.loss_fn(predict, y)
        self.test_accuracy(prediction, y)
        self.test_f1(prediction, y)

        return loss

    def test_epoch_end(self, outputs):
        test_epoch_acc = self.test_accuracy.compute()
        self.test_accuracy.reset()

        loss = sum(outputs) / len(outputs)

        print('test_epoch_accuracy', test_epoch_acc)
        print('test_epoch_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
