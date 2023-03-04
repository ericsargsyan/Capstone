import os
import argparse
from dataflow.utils import read_yaml
from dataflow.utils import mp3_to_wav
from dataflow.importers.mozilla import LanguageImporter, AccentImporter


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        type=str,
                        required=True)
    return parser.parse_args()


name_to_class = {'language_detection':
                            {'MozillaCV': LanguageImporter},
                 'accent_detection':
                            {'MozillaCV': AccentImporter}
                 }

if __name__ == '__main__':
    parser = arg_parser()
    config = read_yaml(parser.config_path)

    if config['convert']['need_to_convert']:
        for convert_data in config['convert']['datasets_to_convert']:
            mp3_to_wav(config['languages'], config['datasets'][convert_data]['source_path'])

    exit()

    for task in config['task']:
        print(f"{'-' * 65}| Processing {task.title()} Data |{'-' * 65}")
        os.makedirs(os.path.join(config['target_dir'], task), exist_ok=True)

        for dataset_name in config['datasets_to_process']:
            dataset_importer = name_to_class[task][dataset_name](config)
            dataset_importer.import_dataset()
