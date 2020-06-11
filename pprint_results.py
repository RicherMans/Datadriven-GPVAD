import argparse
import pandas as pd
import re
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('fmt', default=None,nargs='?')
args = parser.parse_args()

res = {}
root_dir = Path(args.dir)
train_log = root_dir / 'train.log'

config = torch.load(root_dir / 'run_config.pth')
pretrained = config.get('pretrained', None)
# logs
augment = config.get('transforms', [])
label_type = config.get('label_type', 'soft')
model = config.get('model','CRNN')


def get_seg_metrics(line, pointer, seg_type='Segment'):
    res = {}
    while not 'macro-average' in line:
        line = next(pointer).strip()
    while not 'F-measure (F1)' in line:
        line = next(pointer).strip()
    res[f'F1'] = float(line.split()[-2])
    while not 'Precision' in line:
        line = next(pointer).strip()
    res[f'Precision'] = float(line.split()[-2])
    while not 'Recall' in line:
        line = next(pointer).strip()
    res[f'Recall'] = float(line.split()[-2])
    return res


def parse_eval_file(eval_file):
    res = {}
    frame_results = {}
    with open(eval_file, 'r') as rp:
        for line in rp:
            line = line.strip()
            if 'AUC' in line:
                auc = line.split()[-1]
                frame_results['AUC'] = float(auc)
            if 'FER' in line:
                fer = line.split()[-1]
                frame_results['FER'] = float(fer)
            if 'VAD macro' in line:
                f1, pre, rec = re.findall(r"[-+]?\d*\.\d+|\d+",
                                          line)[1:]  # First hit is F1
                frame_results['F1'] = float(f1)
                frame_results['Precision'] = float(pre)
                frame_results['Recall'] = float(rec)
            if "Segment based metrics" in line:
                res['Segment'] = get_seg_metrics(line, rp)
            if 'Event based metrics' in line:
                res['Event'] = get_seg_metrics(line, rp, 'Event')
    res['Frame'] = frame_results
    return res


all_results = []
for f in root_dir.glob('*.txt'):
    eval_dataset = str(f.stem)[11:]
    res = parse_eval_file(f)
    df = pd.DataFrame(res).fillna('')
    df['data'] = eval_dataset
    df['augment'] = ",".join(augment)
    df['pretrained'] = pretrained
    df['label_type'] = label_type
    df['model'] = model
    all_results.append(df)
df = pd.concat(all_results)
if args.fmt == 'csv':
    print(df.to_csv())
else:
    print(df)
