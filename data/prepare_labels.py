import torch
import numpy as np
import argparse
from h5py import File
from pathlib import Path
from loguru import logger
import torch.utils.data as tdata
from tqdm import tqdm
from models import crnn, cnn10
import sys
import csv


class HDF5Dataset(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """
    def __init__(self, h5file: File, transform=None):
        super(HDF5Dataset, self).__init__()
        self._h5file = h5file
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        with File(self._h5file, 'r') as store:
            self._len = len(store)
            self._labels = list(store.keys())
            self.datadim = store[self._labels[0]].shape[-1]

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = File(self._h5file, 'r')
        fname = self._labels[index]
        data = self.dataset[fname][()]
        data = torch.as_tensor(data).float()
        if self._transform:
            data = self._transform(data)
        return data, fname


MODELS = {
    'crnn': {
        'model': crnn,
        'encoder': torch.load('encoders/balanced.pth'),
        'outputdim': 527,
    },
    'gpvb': {
        'model': crnn,
        'encoder': torch.load('encoders/balanced_binary.pth'),
        'outputdim': 2,
    }
}

POOLING = {
    'max': lambda x: np.max(x, axis=-1),
    'mean': lambda x: np.mean(x, axis=-1)
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('-m', '--model', default='crnn', type=str)
    parser.add_argument('-po',
                        '--pool',
                        default='max',
                        choices=POOLING.keys(),
                        type=str)
    parser.add_argument('--pre', '-p', default='pretrained/gpv_f.pth')
    parser.add_argument('hdf5output', type=Path)
    parser.add_argument('csvoutput', type=Path)
    args = parser.parse_args()

    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])

    for k, v in vars(args).items():
        logger.info(f"{k} : {v}")

    model_dict = MODELS[args.model]
    model = model_dict['model'](outputdim=model_dict['outputdim'],
                                pretrained_from=args.pre).to(DEVICE).eval()
    encoder = model_dict['encoder']
    logger.info(model)
    pooling_fun = POOLING[args.pool]

    dataloader = tdata.DataLoader(HDF5Dataset(args.data),
                                  num_workers=4,
                                  batch_size=1)
    speech_class_idx = np.where(encoder.classes_ == 'Speech')[0]
    non_speech_idx = np.arange(len(encoder.classes_))
    non_speech_idx = np.delete(non_speech_idx, speech_class_idx)
    with torch.no_grad(), File(
            args.output, 'w') as store, tqdm(total=len(dataloader)) as pbar, open(args.csvoutput, 'w') as csvfile:
        abs_output_hdf5 = Path(args.output).absolute()
        csvwr = csv.writer(csvfile, delimiter='\t')
        csvwr.writerow(['filename', 'hdf5path'])
        for batch in dataloader:
            x, fname = batch
            fname = fname[0]
            x = x.to(DEVICE)
            if x.shape[1] < 8:
                continue
            clip_pred, time_pred = model(x)
            clip_pred = clip_pred.squeeze(0).to('cpu').numpy()
            time_pred = time_pred.squeeze(0).to('cpu').numpy()
            speech_time_pred = time_pred[..., speech_class_idx].squeeze(-1)
            speech_clip_pred = clip_pred[..., speech_class_idx].squeeze(-1)
            non_speech_clip_pred = clip_pred[..., non_speech_idx]
            non_speech_time_pred = time_pred[..., non_speech_idx]
            non_speech_time_pred = pooling_fun(non_speech_time_pred)
            store[f'{fname}/speech'] = speech_time_pred
            store[f'{fname}/noise'] = non_speech_time_pred
            store[f'{fname}/clipspeech'] = speech_clip_pred
            store[f'{fname}/clipnoise'] = non_speech_clip_pred
            csvwr.writerow([fname, abs_output_hdf5])
            pbar.set_postfix(fname=fname, speechsize=speech_time_pred.shape)
            pbar.update()


if __name__ == "__main__":
    main()
