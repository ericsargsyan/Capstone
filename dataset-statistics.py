import os
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm

path = '/media/capstone/HDD2/source_data/MozillaCV/en/validated.tsv'
en_data = pd.read_table(path)[['path']]

durations = []
bins = np.arange(0, 10, 0.3)

for audio in tqdm(en_data['path']):
    data, samplerate = sf.read(os.path.join('/media/capstone/HDD2/source_data/MozillaCV/en/clips', f'{audio[:-3]}wav'))
    duration = data.shape[0] / samplerate
    durations.append(duration)

plt.hist(durations, bins=bins)
plt.savefig('en_hist_normal.png')
