import torch
import sys
from loguru import logger
from pathlib import Path
from tqdm import tqdm
import utils
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import uuid
import argparse
from models import crnn

SAMPLE_RATE = 22050
EPS = np.spacing(1)
LMS_ARGS = {
    'n_fft': 2048,
    'n_mels': 64,
    'hop_length': int(SAMPLE_RATE * 0.02),
    'win_length': int(SAMPLE_RATE * 0.04)
}
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
DEVICE = torch.device(DEVICE)


def extract_feature(wavefilepath, **kwargs):
    wav, sr = sf.read(wavefilepath, dtype='float32')
    if wav.ndim > 1:
        wav = wav.mean(-1)
    wav = librosa.resample(wav, sr, target_sr=SAMPLE_RATE)
    return np.log(
        librosa.feature.melspectrogram(wav.astype(np.float32), sr, **kwargs) +
        EPS).T


class OnlineLogMelDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, **kwargs):
        super().__init__()
        self.dlist = data_list
        self.kwargs = kwargs

    def __getitem__(self, idx):
        return extract_feature(wavefilepath=self.dlist[idx],
                               **self.kwargs), self.dlist[idx]

    def __len__(self):
        return len(self.dlist)


MODELS = {
    't1': {
        'model': crnn,
        'outputdim': 527,
        'encoder': 'labelencoders/teacher.pth',
        'pretrained': 'teacher1/model.pth',
        'resolution': 0.02
    },
    't2': {
        'model': crnn,
        'outputdim': 527,
        'encoder': 'labelencoders/teacher.pth',
        'pretrained': 'teacher2/model.pth',
        'resolution': 0.02
    },
    'sre': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'sre/model.pth',
        'resolution': 0.02
    },
    'v2': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'vox2/model.pth',
        'resolution': 0.02
    },
    'a2': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'audioset2/model.pth',
        'resolution': 0.02
    },
    'a2_v2': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'audio2_vox2/model.pth',
        'resolution': 0.02
    },
}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-w',
        '--wav',
        help=
        'A single wave/mp3/flac or any other compatible audio file with soundfile.read'
    )
    group.add_argument(
        '-l',
        '--wavlist',
        help=
        'A list of wave or any other compatible audio files. E.g., output of find . -type f -name *.wav > wavlist.txt'
    )
    parser.add_argument('-model', choices=list(MODELS.keys()), default='sre')
    parser.add_argument(
        '--pretrained_dir',
        default='pretrained_models',
        help=
        'Path to downloaded pretrained models directory, (default %(default)s)'
    )
    parser.add_argument('-o',
                        '--output_path',
                        default=None,
                        help='Output folder to save predictions if necessary')
    parser.add_argument('-soft',
                        default=False,
                        action='store_true',
                        help='Outputs soft probabilities.')
    parser.add_argument('-hard',
                        default=False,
                        action='store_true',
                        help='Outputs hard labels as zero-one array.')
    parser.add_argument('-th',
                        '--threshold',
                        default=(0.5, 0.1),
                        type=float,
                        nargs="+")
    args = parser.parse_args()
    pretrained_dir = Path(args.pretrained_dir)
    if not (pretrained_dir.exists() and pretrained_dir.is_dir()):
        logger.error(f"""Pretrained directory {args.pretrained_dir} not found.
Please download the pretrained models from and try again or set --pretrained_dir to your directory."""
                     )
        return
    logger.info("Passed args")
    for k, v in vars(args).items():
        logger.info(f"{k} : {str(v):<10}")
    if args.wavlist:
        wavlist = pd.read_csv(args.wavlist,
                              usecols=[0],
                              header=None,
                              names=['filename'])
        wavlist = wavlist['filename'].values.tolist()
    elif args.wav:
        wavlist = [args.wav]
    dset = OnlineLogMelDataset(wavlist, **LMS_ARGS)
    dloader = torch.utils.data.DataLoader(dset,
                                          batch_size=1,
                                          num_workers=3,
                                          shuffle=False)

    model_kwargs_pack = MODELS[args.model]
    model_resolution = model_kwargs_pack['resolution']
    # Load model from relative path
    model = model_kwargs_pack['model'](
        outputdim=model_kwargs_pack['outputdim'],
        pretrained_from=pretrained_dir /
        model_kwargs_pack['pretrained']).to(DEVICE).eval()
    encoder = torch.load(pretrained_dir / model_kwargs_pack['encoder'])
    logger.trace(model)

    output_dfs = []
    frame_outputs = {}
    threshold = tuple(args.threshold)

    speech_label_idx = np.where('Speech' == encoder.classes_)[0].squeeze()
    # Using only binary thresholding without filter
    if len(threshold) == 1:
        postprocessing_method = utils.binarize
    else:
        postprocessing_method = utils.double_threshold
    with torch.no_grad(), tqdm(total=len(dloader), leave=False,
                               unit='clip') as pbar:
        for feature, filename in dloader:
            feature = torch.as_tensor(feature).to(DEVICE)
            prediction_tag, prediction_time = model(feature)
            prediction_tag = prediction_tag.to('cpu')
            prediction_time = prediction_time.to('cpu')

            if prediction_time is not None:  # Some models do not predict timestamps

                cur_filename = filename[0]  #Remove batchsize
                thresholded_prediction = postprocessing_method(
                    prediction_time, *threshold)
                speech_soft_pred = prediction_time[..., speech_label_idx]
                if args.soft:
                    speech_soft_pred = prediction_time[
                        ..., speech_label_idx].numpy()
                    frame_outputs[cur_filename] = speech_soft_pred[
                        0]  # 1 batch

                if args.hard:
                    speech_hard_pred = thresholded_prediction[...,
                                                              speech_label_idx]
                    frame_outputs[cur_filename] = speech_hard_pred[
                        0]  # 1 batch
                # frame_outputs_hard.append(thresholded_prediction)

                labelled_predictions = utils.decode_with_timestamps(
                    encoder, thresholded_prediction)
                pred_label_df = pd.DataFrame(
                    labelled_predictions[0],
                    columns=['event_label', 'onset', 'offset'])
                if not pred_label_df.empty:
                    pred_label_df['filename'] = cur_filename
                    pred_label_df['onset'] *= model_resolution
                    pred_label_df['offset'] *= model_resolution
                    pbar.set_postfix(labels=','.join(
                        np.unique(pred_label_df['event_label'].values)))
                    pbar.update()
                    output_dfs.append(pred_label_df)

    full_prediction_df = pd.concat(output_dfs).reset_index()
    prediction_df = full_prediction_df[full_prediction_df['event_label'] ==
                                       'Speech']

    if args.output_path:
        args.output_path = Path(args.output_path)
        args.output_path.mkdir(parents=True, exist_ok=True)
        prediction_df.to_csv(args.output_path / 'speech_predictions.tsv',
                             sep='\t',
                             index=False)
        full_prediction_df.to_csv(args.output_path / 'all_predictions.tsv',
                                  sep='\t',
                                  index=False)

        if args.soft or args.hard:
            prefix = 'soft' if args.soft else 'hard'
            with open(args.output_path / f'{prefix}_predictions.txt',
                      'w') as wp:
                np.set_printoptions(suppress=True,
                                    precision=2,
                                    linewidth=np.inf)
                for fname, output in frame_outputs.items():
                    print(f"{fname} {output}", file=wp)
        logger.info(f"Putting results also to dir {args.output_path}")
    if args.soft or args.hard:
        np.set_printoptions(suppress=True, precision=2, linewidth=np.inf)
        for fname, output in frame_outputs.items():
            print(f"{fname} {output}")
    else:
        print(prediction_df.to_markdown(showindex=False))


if __name__ == "__main__":
    main()
