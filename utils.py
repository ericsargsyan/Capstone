import os
import subprocess


def ogg_to_wav(input_file, output_file):
    command = f"ffmpeg -i {input_file} {output_file}"
    subprocess.call(command, shell=True)


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
