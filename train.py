import os
import argparse
from dataflow.utils import read_yaml
from dataset import AudioDataset
from model import AudioModel
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from utils import get_last_version_number
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
    task = config['task']
    model_config = read_yaml(config['model_config_path'])

    dataloader_config = config['dataloader']
    train_path = config['data']['language_detection']['train_path']
    val_path = config['data']['language_detection']['val_path']

    train_dataset = AudioDataset(train_path, config['encodings'][task], task)
    val_dataset = AudioDataset(val_path, config['encodings'][task], task)

    train_dataloader = DataLoader(train_dataset, batch_size=dataloader_config['batch_size'],
                                  shuffle=True, num_workers=config['dataloader']['num_workers'])
    val_dataloader = DataLoader(val_dataset, batch_size=dataloader_config['batch_size'],
                                shuffle=False, num_workers=config['dataloader']['num_workers'])

    model = AudioModel(model_config,
                       config['audio_processor'],
                       config['sr'],
                       max(config['encodings'][task] + 1),
                       config['learning_rate'])

    log_dir_path = config['log_dir'][task]
    version_number = get_last_version_number(log_dir_path)
    logger = TensorBoardLogger(os.path.join(log_dir_path, version_number), name='', version='')
    checkpoints_dir = os.path.join(log_dir_path, version_number, 'checkpoints')

    checkpoint_callback = ModelCheckpoint(save_top_k=5,
                                          filename="{epoch:02d}-{val_epoch_accuracy:.6f}",
                                          dirpath=checkpoints_dir,
                                          monitor='val_epoch_accuracy')

    trainer = Trainer(callbacks=[checkpoint_callback], logger=logger, **config['pl_trainer'])
    trainer.fit(model, train_dataloader, val_dataloader)
