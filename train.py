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
import torch


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
    train_path = config['data'][task]['train_path']
    val_path = config['data'][task]['val_path']

    train_dataset = AudioDataset(train_path, config['encodings'][task], task)
    val_dataset = AudioDataset(val_path, config['encodings'][task], task)

    train_dataloader = DataLoader(train_dataset, batch_size=dataloader_config['batch_size'],
                                  shuffle=True, num_workers=config['dataloader']['num_workers'])
    val_dataloader = DataLoader(val_dataset, batch_size=dataloader_config['batch_size'],
                                shuffle=False, num_workers=config['dataloader']['num_workers'])

    log_dir_path = config['log_dir'][task]
    version_number = get_last_version_number(log_dir_path)
    logger = TensorBoardLogger(os.path.join(log_dir_path, version_number), name='', version='')
    checkpoints_dir = os.path.join(log_dir_path, version_number, 'checkpoints')

    os.makedirs(os.path.join(log_dir_path, version_number), exist_ok=True)
    checkpoint_callback = ModelCheckpoint(save_top_k=5,
                                          filename="{epoch}-{val_epoch_accuracy:.6f}",
                                          dirpath=checkpoints_dir,
                                          monitor='val_epoch_accuracy',
                                          mode='max',
                                          save_last=True
                                          )

    config_copy_command = f"cp {parser.config_path} {os.path.join(log_dir_path, version_number, 'config.yaml')}"
    os.system(config_copy_command)

    # checkpoint = torch.load(config['checkpoint_path'][task.split('_')[0]])
    # model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = AudioModel.load_from_checkpoint(config['checkpoint_path'][task.split('_')[0]],
                                            model_config=model_config,
                                            processor_config=config['audio_processor'],
                                            sr=config['sr'],
                                            number_of_labels=max(config['encodings'][task].values()) + 1,
                                            learning_rate=config['learning_rate'],
                                            encodings=config['encodings'][task],
                                            dataset_statistics=train_dataset.get_statistics(),
                                            include_optimizer=True)

    trainer = Trainer(callbacks=[checkpoint_callback], logger=logger, **config['pl_trainer'])
    trainer.fit(model, train_dataloader, val_dataloader)
