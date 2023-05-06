import soundfile as sf
import numpy as np
from glob import glob
from tqdm import tqdm

path = '/media/capstone/HDD2/data/youtube-hy/*'

for audio in tqdm(glob(path)):
    data, samplerate = sf.read(audio)

    duration = data.shape[0] / 16000

    if duration > 5:
        data = data[:16000 * 5]
    else:
        diff = 5 * 16000 - data.shape[0]
        probability = np.random.rand()
        p, p_1 = int(diff * probability), diff - int(diff * probability)
        data = np.pad(data, pad_width=(p, p_1))

    sf.write(audio, data, samplerate)
