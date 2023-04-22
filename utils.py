import os
from glob import glob


def ogg_to_wav(input_file, output_file):
    command = f"ffmpeg -y -hwaccel cuda -i {input_file} -acodec pcm_s16le -ac 1 -ar 16000 {output_file}"
    os.system(command)


def get_last_version_number(log_dir):
    if not os.path.exists(log_dir):
        return "version_0"

    versions = os.listdir(log_dir)
    versions = [version for version in versions if version[0] != '.']

    if len(versions) == 0:
        return "version_0"

    indexes = [int(version.split('_')[1]) for version in versions]
    max_version_index = max(indexes)

    return f'version_{max_version_index+1}'


def get_best_checkpoint(base_path, check_path):
    check_abs_path = os.path.join(base_path, check_path, 'checkpoints')
    checkpoint_files = glob(os.path.join(check_abs_path, '*.ckpt'))

    accuracies = [float(checkpoint.split('=')[2][:-5]) for checkpoint in checkpoint_files]
    checkpoint_path = checkpoint_files[accuracies.index(max(accuracies))]

    return checkpoint_path
