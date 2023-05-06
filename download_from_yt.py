import pytube
from tqdm import tqdm
import os
from glob import glob

data_path = '/media/capstone/HDD2/source_data/youtube_links2/video_links.txt'

link_names = [i for i in glob(f'{data_path}{os.sep}*')]

with open('/media/capstone/HDD2/source_data/youtube_links2/PL_RnEKSdOwG0s8qm9hNMSLHj7cVnWiBfG.txt', 'r') as f:
    links = [line.strip() for line in f]

download_path = '/media/capstone/HDD2/source_data/youtube_videos2'

for link in tqdm(links):
    yt = pytube.YouTube(link, use_oauth=True, allow_oauth_cache=True)
    audio = yt.streams.filter(only_audio=True).first()
    audio.download(download_path)
