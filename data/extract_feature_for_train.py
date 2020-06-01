#!/usr/bin/env python3
import argparse
import librosa
from tqdm import tqdm
import io
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np
import soundfile as sf
from pypeln import process as pr
import h5py
import gzip

parser = argparse.ArgumentParser()
parser.add_argument('input_csv')
parser.add_argument('-o', '--output', type=str, required=True)
parser.add_argument('-l',
                    '--label',
                    type=str,
                    required=True,
                    help="Output label(chunked)")
parser.add_argument('-s',
                    '--size',
                    type=float,
                    default=10,
                    help="Length of each segment")
parser.add_argument('-t',
                    '--threshold',
                    type=float,
                    default=1,
                    help='Do not save files less than -t seconds',
                    metavar='s')
parser.add_argument('-c', type=int, default=4)
parser.add_argument('-sr', type=int, default=22050)
parser.add_argument('-col',
                    default='filename',
                    type=str,
                    help='Column to search for audio files')
parser.add_argument('-cmn', default=False, action='store_true')
parser.add_argument('-cvn', default=False, action='store_true')
parser.add_argument('-winlen',
                    default=40,
                    type=float,
                    help='FFT duration in ms')
parser.add_argument('-hoplen',
                    default=20,
                    type=float,
                    help='hop duration in ms')

parser.add_argument('-n_mels', default=64, type=int)
ARGS = parser.parse_args()

DF = pd.read_csv(ARGS.input_csv, usecols=[0], sep=' ')

MEL_ARGS = {
    'n_mels': ARGS.n_mels,
    'n_fft': 2048,
    'hop_length': int(ARGS.sr * ARGS.hoplen / 1000),
    'win_length': int(ARGS.sr * ARGS.winlen / 1000)
}

EPS = np.spacing(1)
DURATION_CHUNK = ARGS.size / (ARGS.hoplen / 1000)
THRESHOLD = ARGS.threshold / (ARGS.hoplen / 1000)


def extract_feature(fname):
    # def extract_feature(fname, segfname, start, end, nseg):
    """extract_feature
    Extracts a log mel spectrogram feature from a filename, currently supports two filetypes:

    1. Wave
    2. Gzipped wave

    :param fname: filepath to the file to extract
    """
    pospath = Path(fname)
    ext = pospath.suffix
    try:
        if ext == '.gz':
            with gzip.open(fname, 'rb') as gzipped_wav:
                y, sr = sf.read(io.BytesIO(gzipped_wav.read()),
                                dtype='float32')
                # Multiple channels, reduce
                if y.ndim == 2:
                    y = y.mean(1)
                y = librosa.resample(y, sr, ARGS.sr)
        elif ext in ('.wav', '.flac'):
            y, sr = sf.read(fname, dtype='float32')
            if y.ndim > 1:
                y = y.mean(1)
            y = librosa.resample(y, sr, ARGS.sr)
    except Exception as e:
        # Exception usually happens because some data has 6 channels , which librosa cant handle
        logger.error(e)
        logger.error(fname)
        raise
    fname = pospath.name
    feat = np.log(librosa.feature.melspectrogram(y, **MEL_ARGS) + EPS).T
    start_range = np.arange(0, feat.shape[0], DURATION_CHUNK, dtype=int)
    end_range = (start_range + DURATION_CHUNK).astype(int)
    end_range[-1] = feat.shape[0]
    for nseg, (start_time, end_time) in enumerate(zip(start_range, end_range)):
        seg = feat[start_time:end_time]
        if end_time - start_time < THRESHOLD:
            # Dont save
            continue
        yield fname, seg, nseg


with h5py.File(ARGS.output, 'w') as store, tqdm() as pbar, open(ARGS.label,'w') as output_csv:
    output_csv.write(f"filename hdf5path\n") #write header
    hdf5_path = Path(ARGS.output).absolute()
    for fname, feat, nseg in pr.flat_map(extract_feature,
                                         DF['filename'].unique(),
                                         workers=ARGS.c,
                                         maxsize=ARGS.c * 2):
        new_fname = f"{Path(fname).stem}_{nseg:05d}{Path(fname).suffix}"
        store[new_fname] = feat
        output_csv.write(f"{new_fname} {hdf5_path}\n")
        pbar.set_postfix(stored=new_fname, shape=feat.shape)
        pbar.update()
