# Data driven VAD
Repository for the work [Data-driven VAD](www.localhost.com)


![Framework](figs/data_driven_framework.png)

![Speech](figs/sample_speech.png)

![Background](figs/sample_background.png)

![Speech](figs/sample_speech.png)

## Usage

We provide most of our pretrained models in this repository, including:

1. Both teachers (T_1, T_2)
2. Unbalanced audioset pretrained model
3. Voxceleb 2 pretrained model
4. Our best submission (SRE trained)

To download and run evaluation just do:

```bash
git clone https://github.com/RicherMans/Datadriven-VAD
wget ddata
unzip pretrained_models.zip
python3 forward.py -w example/example.wav

```

### Predicting voice activity

We support single file and filelist-batching in our script. 
Obtaining VAD predictions is easy:

```bash
python3 forward.py -w /mnt/lustre/sjtu/users/sw121/import/CN-Celeb/data/id00802/speech-03-027.wav
```

Or if one prefers to do that batch_wise, first prepare a filelist:
`find . -type f -name *.wav > wavlist.txt'`
And then just run:
```bash
python3 forward.py -l wavlist
```


#### Extra parameters

* `-model` adjusts the pretrained model. Can be one of `t1,t2,v2,a2,a2_v2,sre`. Refer to the paper for each respective model. By default we use `sre`.
* `-soft` instead of predicting human-readable timestamps, the model is now outputting the raw probabilities.
* `-hard` instead of predicting human-readable timestamps, the model is now outputting the post-processed 0-1 flags indicating speech. Please note this is different from the paper, which thresholded the soft probabilities without post-processing.
* `-th` adjusts the threshold. If a single threshold is passed (e.g., `-th 0.5`), we utilize simple binearization. Otherwise use the default double threshold with `-th 0.5 0.1`.
* `-o` outputs the results into a new folder.


## Training from scratch

If you intend to rerun our work, prepare some data.
