import os
import argparse
from dataflow.utils import read_yaml
from dataset import AccentDataset, LanguageDataset
from model import AudioModel
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        type=str,
                        required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parser = arg_parser()
    config = read_yaml(parser.config_path)
    dataloader_config = config['dataloader']

    train_path = config['data']['language_detection']['train_path']
    val_path = config['data']['language_detection']['val_path']

    train_dataset = LanguageDataset(train_path)
    val_dataset = LanguageDataset(val_path)

    train_dataloader = DataLoader(train_dataset, batch_size=dataloader_config['batch_size'],
                                                 shuffle=True, num_workers=config['dataloader']['num_workers'])
    val_dataloader = DataLoader(val_dataset, batch_size=dataloader_config['batch_size'],
                                             shuffle=False, num_workers=config['dataloader']['num_workers'])

    model = AudioModel(3, config['learning_rate'])

    trainer = Trainer(max_epochs=5)
    trainer.fit(model, train_dataloader, val_dataloader)
