#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import datetime

import uuid
import fire
from pathlib import Path

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from ignite.contrib.handlers import ProgressBar, param_scheduler
from ignite.engine import (Engine, Events)
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy, RunningAverage, Precision, Recall
from ignite.utils import convert_tensor
from tabulate import tabulate
from h5py import File
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support

import dataset
import models
import utils
import metrics
import losses

DEVICE = 'cpu'
if torch.cuda.is_available(
) and 'SLURM_JOB_PARTITION' in os.environ and 'gpu' in os.environ[
        'SLURM_JOB_PARTITION']:
    DEVICE = 'cuda'
    # Without results are slightly inconsistent
    torch.backends.cudnn.deterministic = True
DEVICE = torch.device(DEVICE)


class Runner(object):
    """Main class to run experiments with e.g., train and evaluate"""
    def __init__(self, seed=42):
        """__init__

        :param config: YAML config file
        :param **kwargs: Overwrite of yaml config
        """
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

    @staticmethod
    def _forward(model, batch):
        inputs, targets_time, targets_clip, filenames, lengths = batch
        inputs = convert_tensor(inputs, device=DEVICE, non_blocking=True)
        targets_time = convert_tensor(targets_time,
                                      device=DEVICE,
                                      non_blocking=True)
        targets_clip = convert_tensor(targets_clip,
                                      device=DEVICE,
                                      non_blocking=True)
        clip_level_output, frame_level_output = model(inputs)
        return clip_level_output, frame_level_output, targets_time, targets_clip, lengths

    @staticmethod
    def _negative_loss(engine):
        return -engine.state.metrics['Loss']

    def train(self, config, **kwargs):
        """Trains a given model specified in the config file or passed as the --model parameter.
        All options in the config file can be overwritten as needed by passing --PARAM
        Options with variable lengths ( e.g., kwargs can be passed by --PARAM '{"PARAM1":VAR1, "PARAM2":VAR2}'

        :param config: yaml config file
        :param **kwargs: parameters to overwrite yaml config
        """

        config_parameters = utils.parse_config_or_kwargs(config, **kwargs)
        outputdir = os.path.join(
            config_parameters['outputpath'], config_parameters['model'],
            "{}_{}".format(
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%m'),
                uuid.uuid1().hex))
        # Early init because of creating dir
        checkpoint_handler = ModelCheckpoint(
            outputdir,
            'run',
            n_saved=1,
            require_empty=False,
            create_dir=True,
            score_function=self._negative_loss,
            score_name='loss')
        logger = utils.getfile_outlogger(os.path.join(outputdir, 'train.log'))
        logger.info("Storing files in {}".format(outputdir))
        # utils.pprint_dict
        utils.pprint_dict(config_parameters, logger.info)
        logger.info("Running on device {}".format(DEVICE))
        with File(config_parameters['label'], 'r') as label_store:
            filenames = list(label_store.keys())

        # These labels are useless, only for mode == stratified
        train_fnames, cv_fnames = utils.split_train_cv(
            filenames, **config_parameters['data_args'])

        transform = utils.parse_transforms(config_parameters['transforms'])
        torch.save(config_parameters, os.path.join(outputdir,
                                                   'run_config.pth'))
        logger.info("Transforms:")
        utils.pprint_dict(transform, logger.info, formatter='pretty')
        assert len(cv_fnames) > 0, "Fraction a bit too large?"

        trainloader = dataset.getdataloader(
            h5file=config_parameters['data'],
            h5label=config_parameters['label'],
            fnames=train_fnames,
            transform=transform,
            batch_size=config_parameters['batch_size'],
            num_workers=config_parameters['num_workers'],
            shuffle=True)

        cvdataloader = dataset.getdataloader(
            h5file=config_parameters['data'],
            h5label=config_parameters['label'],
            fnames=cv_fnames,
            transform=None,
            shuffle=False,
            batch_size=config_parameters['batch_size'],
            num_workers=config_parameters['num_workers'])
        model = getattr(models, config_parameters['model'],
                        'CRNN')(inputdim=trainloader.dataset.datadim,
                                outputdim=2,
                                **config_parameters['model_args'])
        if 'pretrained' in config_parameters and config_parameters[
                'pretrained'] is not None:
            model_dump = torch.load(config_parameters['pretrained'],
                                    map_location='cpu')
            model_state = model.state_dict()
            pretrained_state = {
                k: v
                for k, v in model_dump.items()
                if k in model_state and v.size() == model_state[k].size()
            }
            model_state.update(pretrained_state)
            model.load_state_dict(model_state)
            logger.info("Loading pretrained model {}".format(
                config_parameters['pretrained']))

        model = model.to(DEVICE)
        if config_parameters['optimizer'] == 'AdaBound':
            try:
                import adabound
                optimizer = adabound.AdaBound(
                    model.parameters(), **config_parameters['optimizer_args'])
            except ImportError:
                config_parameters['optimizer'] = 'Adam'
                config_parameters['optimizer_args'] = {}
        else:
            optimizer = getattr(
                torch.optim,
                config_parameters['optimizer'],
            )(model.parameters(), **config_parameters['optimizer_args'])

        utils.pprint_dict(optimizer, logger.info, formatter='pretty')
        utils.pprint_dict(model, logger.info, formatter='pretty')
        if DEVICE.type != 'cpu' and torch.cuda.device_count() > 1:
            logger.info("Using {} GPUs!".format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model)
        criterion = getattr(losses, config_parameters['loss'])().to(DEVICE)

        def _train_batch(_, batch):
            model.train()
            with torch.enable_grad():
                optimizer.zero_grad()
                output = self._forward(
                    model, batch)  # output is tuple (clip, frame, target)
                loss = criterion(*output)
                loss.backward()
                # Single loss
                optimizer.step()
                return loss.item()

        def _inference(_, batch):
            model.eval()
            with torch.no_grad():
                return self._forward(model, batch)

        def thresholded_output_transform(output):
            # Output is (clip, frame, target, lengths)
            _, y_pred, y, y_clip, length = output
            batchsize, timesteps, ndim = y.shape
            idxs = torch.arange(timesteps,
                                device='cpu').repeat(batchsize).view(
                                    batchsize, timesteps)
            mask = (idxs < length.view(-1, 1)).to(y.device)
            y = y * mask.unsqueeze(-1)
            y_pred = torch.round(y_pred)
            y = torch.round(y)
            return y_pred, y

        metrics = {
            'Loss': losses.Loss(
                criterion),  #reimplementation of Loss, supports 3 way loss 
            'Precision': Precision(thresholded_output_transform),
            'Recall': Recall(thresholded_output_transform),
            'Accuracy': Accuracy(thresholded_output_transform),
        }
        train_engine = Engine(_train_batch)
        inference_engine = Engine(_inference)
        for name, metric in metrics.items():
            metric.attach(inference_engine, name)

        def compute_metrics(engine):
            inference_engine.run(cvdataloader)
            results = inference_engine.state.metrics
            output_str_list = [
                "Validation Results - Epoch : {:<5}".format(engine.state.epoch)
            ]
            for metric in metrics:
                output_str_list.append("{} {:<5.2f}".format(
                    metric, results[metric]))
            logger.info(" ".join(output_str_list))
            pbar.n = pbar.last_print_n = 0

        pbar = ProgressBar(persist=False)
        pbar.attach(train_engine)

        train_engine.add_event_handler(Events.ITERATION_COMPLETED(every=2000),
                                       compute_metrics)
        train_engine.add_event_handler(Events.EPOCH_COMPLETED, compute_metrics)

        early_stop_handler = EarlyStopping(
            patience=config_parameters['early_stop'],
            score_function=self._negative_loss,
            trainer=train_engine)
        inference_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                           early_stop_handler)
        inference_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                           checkpoint_handler, {
                                               'model': model,
                                           })

        train_engine.run(trainloader, max_epochs=config_parameters['epochs'])
        return outputdir

    def train_evaluate(self, config, test_data, test_label, **kwargs):
        experiment_path = self.train(config, **kwargs)
        import h5py
        # Get the output time-ratio factor from the model
        model_parameters = torch.load(
            Path("{}").glob("run_model*".format(experiment_path))[0],
            map_location=lambda storage, loc: storage)
        config_param = torch.load(Path('{}').glob(
            "run_config*".format(experiment_path))[0],
                                  map_location=lambda storage, loc: storage)
        encoder = torch.load('labelencoders/vad.pth')
        # Dummy to calculate the pooling factor a bit dynamic
        with h5py.File(test_data, 'r') as store:
            timedim, datadim = next(iter(store.values())).shape
        model = getattr(models,
                        config_param['model'])(inputdim=datadim,
                                               outputdim=len(encoder.classes_),
                                               **config_param['model_args'])
        model.load_state_dict(model_parameters)
        dummy = torch.randn(1, timedim, datadim)
        _, time_out = model(dummy)
        time_ratio = max(0.02, 0.02 * np.round(timedim / time_out.shape[1]))
        # Parse for evaluation and update original values such as
        # --data
        # --label
        config_parameters = utils.parse_config_or_kwargs(config, **kwargs)
        threshold = config_parameters.get('threshold', None)
        postprocessing = config_parameters.get('postprocessing', 'double')
        window_size = config_parameters.get('window_size', None)
        self.evaluate(experiment_path,
                      label=test_label,
                      data=test_data,
                      time_ratio=time_ratio,
                      postprocessing=postprocessing,
                      threshold=threshold,
                      window_size=window_size)

    def predict_time(
            self,
            experiment_path,
            output_h5,
            rfac=2,  # Resultuion upscale fator
            **kwargs):  # overwrite --data

        experiment_path = Path(experiment_path)
        config = torch.load(next(Path(experiment_path).glob("run_config*")),
                            map_location=lambda storage, loc: storage)
        logger = utils.getfile_outlogger(None)
        # Use previous config, but update data such as kwargs
        config_parameters = dict(config, **kwargs)
        # Default columns to search for in data
        model_parameters = torch.load(
            next(Path(experiment_path).glob("run_model*")),
            map_location=lambda storage, loc: storage)
        encoder = torch.load('labelencoders/vad.pth')
        data = config_parameters['data']
        dset = dataset.EvalH5Dataset(data)
        dataloader = torch.utils.data.DataLoader(dset,
                                                 batch_size=1,
                                                 num_workers=4,
                                                 shuffle=False)

        model = getattr(models, config_parameters['model'])(
            inputdim=dataloader.dataset.datadim,
            outputdim=len(encoder.classes_),
            **config_parameters['model_args'])

        model.load_state_dict(model_parameters)
        model = model.to(DEVICE).eval()

        ## VAD preprocessing data
        logger.trace(model)

        output_dfs = []

        speech_label_idx = np.where('Speech' == encoder.classes_)[0].squeeze()
        non_speech_idx = np.arange(len(encoder.classes_))
        non_speech_idx = np.delete(non_speech_idx, speech_label_idx)
        speech_frame_predictions, speech_frame_prob_predictions = [], []
        with torch.no_grad(), tqdm(total=len(dataloader),
                                   leave=False,
                                   unit='clip') as pbar, File(output_h5,
                                                              'w') as store:
            for feature, filename in dataloader:
                feature = torch.as_tensor(feature).to(DEVICE)
                filename = Path(filename[0]).stem
                batch, time, dim = feature.shape
                # PANNS output a dict instead of 2 values
                prediction_tag, prediction_time = model(feature,
                                                        upsample=False)
                prediction_tag = prediction_tag.to('cpu')
                prediction_time = torch.nn.functional.interpolate(
                    prediction_time.transpose(1, 2),
                    int(time * rfac),
                    mode='linear',
                    align_corners=False).transpose(1, 2)
                prediction_time = prediction_time.to('cpu').squeeze(0)
                speech_label_pred = prediction_time[
                    ..., speech_label_idx].squeeze(-1)
                noise_label_pred = prediction_time[...,
                                                   non_speech_idx].squeeze(-1)
                store[f'{filename}/speech'] = speech_label_pred
                store[f'{filename}/noise'] = noise_label_pred
                pbar.set_postfix(time=time,
                                 fname=filename,
                                 speech=speech_label_pred.shape,
                                 noise=noise_label_pred.shape)
                pbar.update()

    def predict_clip(self, experiment_path, output_csv, thres=0.5,
                     **kwargs):  # overwrite --data
        import h5py
        from sklearn.preprocessing import binarize
        from tqdm import tqdm
        config = torch.load(list(Path(experiment_path).glob("run_config*"))[0],
                            map_location=lambda storage, loc: storage)
        config_parameters = dict(config, **kwargs)
        model_parameters = torch.load(
            list(Path(experiment_path).glob("run_model*"))[0],
            map_location=lambda storage, loc: storage)
        encoder = torch.load('labelencoders/vad.pth')

        predictions = []
        with h5py.File(config_parameters['data'],
                       'r') as input_store, torch.no_grad(), tqdm(
                           total=len(input_store)) as pbar:
            inputdim = next(iter(input_store.values())).shape[-1]
            model = getattr(models, config_parameters['model'])(
                inputdim=inputdim,
                outputdim=len(encoder.classes_),
                **config_parameters['model_args'])
            model.load_state_dict(model_parameters)
            model = model.to(DEVICE).eval()
            for fname, sample in input_store.items():
                if sample.ndim > 1:  # Global mean and Global_var might also be there
                    sample = torch.as_tensor(sample[()]).unsqueeze(0).to(
                        DEVICE)  # batch + channel
                    decision, _ = model(sample)
                    decision = binarize(decision.to('cpu'), threshold=thres)
                    pred_labels = encoder.inverse_transform(decision)[0]
                    pbar.set_postfix(labels=pred_labels, file=fname)
                    if len(pred_labels) > 0:
                        predictions.append({
                            'filename':
                            fname,
                            'event_labels':
                            ",".join(pred_labels)
                        })
                pbar.update()

        df = pd.DataFrame(predictions)
        df.to_csv(output_csv, sep='\t', index=False)

    def evaluate(self,
                 experiment_path: Path,
                 task: str = 'aurora_clean',
                 model_resolution=0.02,
                 time_resolution=0.02,
                 threshold=(0.5, 0.1),
                 **kwargs):
        EVALUATION_DATA = {
            'aurora_clean': {
                'data': 'data/evaluation/hdf5/aurora_clean.h5',
                'label': 'data/evaluation/labels/aurora_clean_labels.tsv',
            },
            'aurora_noisy': {
                'data': 'data/evaluation/hdf5/aurora_noisy.h5',
                'label': 'data/evaluation/labels/aurora_noisy_labels.tsv'
            },
            'dcase18': {
                'data': 'data/evaluation/hdf5/dcase18.h5',
                'label': 'data/evaluation/labels/dcase18.tsv',
            },
        }
        assert task in EVALUATION_DATA, f"--task {'|'.join(list(EVALUATION_DATA.keys()))}"
        experiment_path = Path(experiment_path)
        config = torch.load(next(Path(experiment_path).glob("run_config*")),
                            map_location=lambda storage, loc: storage)
        logger = utils.getfile_outlogger(None)
        # Use previous config, but update data such as kwargs
        config_parameters = dict(config, **kwargs)
        # Default columns to search for in data
        model_parameters = torch.load(
            next(Path(experiment_path).glob("run_model*")),
            map_location=lambda storage, loc: storage)
        encoder = torch.load('labelencoders/vad.pth')
        data = EVALUATION_DATA[task]['data']
        label_df = pd.read_csv(EVALUATION_DATA[task]['label'], sep='\s+')
        label_df['filename'] = label_df['filename'].apply(
            lambda x: Path(x).name)
        logger.info(f"Label_df shape is {label_df.shape}")

        dset = dataset.EvalH5Dataset(data)

        dataloader = torch.utils.data.DataLoader(dset,
                                                 batch_size=1,
                                                 num_workers=8,
                                                 shuffle=False)

        model = getattr(models, config_parameters['model'])(
            inputdim=dataloader.dataset.datadim,
            outputdim=len(encoder.classes_),
            **config_parameters['model_args'])

        model.load_state_dict(model_parameters)
        model = model.to(DEVICE).eval()

        ## VAD preprocessing data
        vad_label_helper_df = label_df.copy()
        vad_label_helper_df['onset'] = np.ceil(vad_label_helper_df['onset'] /
                                               model_resolution).astype(int)
        vad_label_helper_df['offset'] = np.ceil(vad_label_helper_df['offset'] /
                                                model_resolution).astype(int)

        vad_label_helper_df = vad_label_helper_df.groupby(['filename']).agg({
            'onset':
            tuple,
            'offset':
            tuple,
            'event_label':
            tuple
        }).reset_index()
        logger.trace(model)

        output_dfs = []

        speech_label_idx = np.where('Speech' == encoder.classes_)[0].squeeze()
        speech_frame_predictions, speech_frame_ground_truth, speech_frame_prob_predictions = [], [],[]
        # Using only binary thresholding without filter
        if len(threshold) == 1:
            postprocessing_method = utils.threshold
        else:
            postprocessing_method = utils.double_threshold
        with torch.no_grad(), tqdm(total=len(dataloader),
                                   leave=False,
                                   unit='clip') as pbar:
            for feature, filename in dataloader:
                feature = torch.as_tensor(feature).to(DEVICE)
                # PANNS output a dict instead of 2 values
                prediction_tag, prediction_time = model(feature)
                prediction_tag = prediction_tag.to('cpu')
                prediction_time = prediction_time.to('cpu')

                if prediction_time is not None:  # Some models do not predict timestamps

                    cur_filename = filename[0]

                    thresholded_prediction = postprocessing_method(
                        prediction_time, *threshold)

                    ## VAD predictions
                    speech_frame_prob_predictions.append(
                        prediction_time[..., speech_label_idx].squeeze())
                    ### Thresholded speech predictions
                    speech_prediction = thresholded_prediction[
                        ..., speech_label_idx].squeeze()
                    speech_frame_predictions.append(speech_prediction)
                    targets = vad_label_helper_df[
                        vad_label_helper_df['filename'] == cur_filename][[
                            'onset', 'offset'
                        ]].values[0]
                    target_arr = np.zeros_like(speech_prediction)
                    for start, end in zip(*targets):
                        target_arr[start:end] = 1
                    speech_frame_ground_truth.append(target_arr)

                    #### SED predictions

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

        full_prediction_df = pd.concat(output_dfs)
        prediction_df = full_prediction_df[full_prediction_df['event_label'] ==
                                           'Speech']
        assert set(['onset', 'offset', 'filename', 'event_label'
                    ]).issubset(prediction_df.columns), "Format is wrong"
        assert set(['onset', 'offset', 'filename', 'event_label'
                    ]).issubset(label_df.columns), "Format is wrong"
        logger.info("Calculating VAD measures ... ")
        speech_frame_ground_truth = np.concatenate(speech_frame_ground_truth,
                                                   axis=0)
        speech_frame_predictions = np.concatenate(speech_frame_predictions,
                                                  axis=0)
        speech_frame_prob_predictions = np.concatenate(
            speech_frame_prob_predictions, axis=0)

        vad_results = []
        tn, fp, fn, tp = confusion_matrix(speech_frame_ground_truth,
                                          speech_frame_predictions).ravel()
        fer = 100 * ((fp + fn) / len(speech_frame_ground_truth))

        auc = roc_auc_score(speech_frame_ground_truth,
                            speech_frame_prob_predictions) * 100
        for avgtype in ('micro', 'macro', 'binary'):
            precision, recall, f1, _ = precision_recall_fscore_support(
                speech_frame_ground_truth,
                speech_frame_predictions,
                average=avgtype)
            vad_results.append(
                (avgtype, 100 * precision, 100 * recall, 100 * f1))

        logger.info("Calculating segment based metric .. ")
        # Change order just for better printing in file
        prediction_df = prediction_df[[
            'filename', 'onset', 'offset', 'event_label'
        ]]
        metric = metrics.segment_based_evaluation_df(
            label_df, prediction_df, time_resolution=time_resolution)
        logger.info("Calculating event based metric .. ")
        event_metric = metrics.event_based_evaluation_df(
            label_df, prediction_df)

        prediction_df.to_csv(experiment_path /
                             f'speech_predictions_{task}.tsv',
                             sep='\t',
                             index=False)
        full_prediction_df.to_csv(experiment_path / f'predictions_{task}.tsv',
                                  sep='\t',
                                  index=False)
        with open(experiment_path / f'evaluation_{task}.txt', 'w') as fp:
            for k, v in config_parameters.items():
                print(f"{k}:{v}", file=fp)
            print(metric, file=fp)
            print(event_metric, file=fp)
            for avgtype, precision, recall, f1 in vad_results:
                print(
                    f"VAD {avgtype} F1: {f1:<10.3f} {precision:<10.3f} Recall: {recall:<10.3f}",
                    file=fp)
            print(f"FER: {fer:.2f}", file=fp)
            print(f"AUC: {auc:.2f}", file=fp)
        logger.info(f"Results are at {experiment_path}")
        for avgtype, precision, recall, f1 in vad_results:
            print(
                f"VAD {avgtype:<10} F1: {f1:<10.3f} Pre: {precision:<10.3f} Recall: {recall:<10.3f}"
            )
        print(f"FER: {fer:.2f}")
        print(f"AUC: {auc:.2f}")
        print(event_metric)
        print(metric)


if __name__ == "__main__":
    fire.Fire(Runner)
