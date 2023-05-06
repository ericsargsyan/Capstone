import os
from tqdm import tqdm


class YoutubeImporter:
    def __init__(self, config):
        self.source_path = config['datasets']['Youtube']['source_path']
        self.wav_paths = config['datasets']['Youtube']['wav_paths']
        os.makedirs(self.wav_paths, exist_ok=True)

    def import_dataset(self):
        audio_names = os.listdir(self.source_path)

        final_destination = '/media/capstone/HDD2/data/youtube-hy2'

        # Dont forget to change section name with the maximum index
        section = 1

        for audio_name in tqdm(audio_names):
            audio_source_path = os.path.join(self.source_path, audio_name)
            output_path = os.path.join(self.wav_paths, f'{audio_name[:-3]}wav')

            command_wav = f'ffmpeg -y -hwaccel cuda -i "{audio_source_path}" -acodec pcm_s16le -ac 1 -ar 16000 "{output_path}"'
            os.system(command_wav)

            command_5sec = f'ffmpeg -i "{output_path}" -f segment -segment_time 5 -c copy {os.path.join(final_destination, f"{section:02d}_%03d.wav")}'
            os.system(command_5sec)

            section += 1
