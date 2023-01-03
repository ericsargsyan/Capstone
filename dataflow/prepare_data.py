import os
import argparse
from dataflow.utils import read_yaml
from dataflow.importers.language.MozillaCV import MozillaCVLImporter
from dataflow.importers.accent.MozillaCV import MozillaCVAImporter


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        type=str,
                        required=True)
    return parser.parse_args()


name_to_class = {'language_detection':
                     {'MozillaCV': MozillaCVLImporter},
                 'accent_detection':
                     {'MozillaCV': MozillaCVAImporter}
                 }

if __name__ == '__main__':
    parser = arg_parser()
    config = read_yaml(parser.config_path)
    datasets_to_process = config['datasets_to_process']

    for task in config['task']:
        print(f'--------------------------------------------------------------\t'
              f'Processing {task.title()} Data'
              f'\t--------------------------------------------------------------')
        os.makedirs(os.path.join(config['target_dir'], task), exist_ok=True)

        for dataset_name in config['datasets_to_process']:
            dataset_importer = name_to_class[task][dataset_name](config)
            dataset_importer.import_dataset()
