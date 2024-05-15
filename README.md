[English](README.md)|[简体中文](README_cn.md)
# Audio Classification
This repository contains the implementation (in PyTorch) of some models for Sound/Audio Classification.

MFCC + TDNN (Mel Frequency Cepstral Coefficents, Time-Delay Neural Network)

## Dataset
[DCASE2020](https://zenodo.org/record/3678171)
  : ./ackit/data_utils/soundreader2020.py

[COUGHVID2021](https://doi.org/10.5281/zenodo.4048311)
  : ./ackit/data_utils/soundexplore.ipynb
  : ./coughvid_reader.py

[covid19-cough](https://github.com/covid19-cough/dataset)
  : ../SoundDL-CoughVID/covid19_explore

The procedure of preprocessing see in soundexplore.ipynb


This dataset contains 20000 audio pieces, each lasting 10 seconds, and a total of 7 categories of machines, totaling 23 different machines.

# Experiments Set

## dsptcls
### Dataset-Pretrain-Classifiers
#### cnn: CNNCls
set1
- input: ./datacreate.ipynb, preprocessing from [mariostrbac](https://github.com/mariostrbac/environmental-sound-classification)
- 2.95s
- run: ./ackit/dsptcls.py

set2
- input: ./datacreate.ipynb, dcase2020 [mariostrbac](https://github.com/mariostrbac/environmental-sound-classification)
- 2.95s
- run: ./ackit/dsptcls.py

set3 Failed
- input: ./datacreate.ipynb, reference: [COVID-19 Screening From Audio | Part 2](https://www.kaggle.com/code/nasrulhakim86/covid-19-screening-from-audio-part-2/notebook)
- dataloader: reference: [mariostrbac](https://github.com/mariostrbac/environmental-sound-classification)

set4
- dataset: covid19 [covid19-cough](https://github.com/covid19-cough/dataset)
- trained with fold10 cycle: not good
- loss: CrossEntropy

set5
- dataset: covid19
- trained with fold1 not cycle: well
- loss: CrossEntropy

set6
- dataset: covid19
- trained with fold1 not cycle: not good
- loss: FocalLoss

## TDNN - coughvid2021
- trainer jupyter 1: ./ackit/coughcls_tdnn.ipynb
- trainer jupyter 2: ./ackit/coughcls_tdnn_focalloss.ipynb
- trainer py: ./ackit/trainer_tdnn.py
- model: ./ackit/models/tdnn.py
- dataset: ./ackit/data_utils/coughvid_reader.py
- dataset: ./datasets/waveinfo_annotation.csv

## CNN - dcase2020
- models: ./ackit/models/cnn_classifier.py
- pretrain_model: ./runs/VAE/model_epoch_12/model_epoch12.pth
- config: ./configs/autoencoder.yaml
- result: ./runs/vae_cnncls/202404181142_ptvae/
- accuracy, precision, recall: ./ackit/utils/plotter.py calc_accuracy(pred_matrix, label_vec, save_path)

### run train
```text
trainer = TrainerEncoder(istrain=True, isdemo=False)
trainer.train_classifier()

>>python trainer_ConvEncoder.py
```

### run t-SNE
```text
trainer = TrainerEncoder(istrain=False, isdemo=False)
trainer.plot_reduction()

>>python trainer_ConvEncoder.py
```
see plot_reduction(self, resume_path="202404181142_ptvae") about how to construct the input of t-SNE

and ./ackit/utils/plotter.py plot_TSNE(embd, names, save_path)

### run Heatmap
procedure same as t-SNE
```text
trainer = TrainerEncoder(istrain=False, isdemo=False)
trainer.plot_heatmap()

>>python trainer_ConvEncoder.py
```
see ./ackit/utils/plotter.py plot_heatmap(pred_matrix, label_vec, savepath)

