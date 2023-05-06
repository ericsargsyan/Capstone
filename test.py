from dataset import AudioDataset
from model import AudioModel
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import argparse
from dataflow.utils import read_yaml
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

    test_dataset = AudioDataset(config['data'][task]['test_path'], config['encodings'][task], task)
    test_dataloader = DataLoader(test_dataset, batch_size=config['dataloader']['batch_size'],
                                 shuffle=False, num_workers=config['dataloader']['num_workers'])

    checkpoint_path = config['checkpoint_path'][task.split('_')[0]]

    model = AudioModel(model_config=read_yaml('./configs/model/net.yaml'),
                       processor_config=config['audio_processor'],
                       sr=config['sr'],
                       number_of_labels=max(config['encodings'][task].values()) + 1,
                       learning_rate=config['learning_rate'],
                       encodings=config['encodings'][task],
                       dataset_statistics=None)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    trainer = Trainer(**config['pl_trainer'])
    trainer.test(model, dataloaders=test_dataloader)
