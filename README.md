[English](README.md)|[简体中文](README_cn.md)
# Audio Classification
This repository contains the implementation (in PyTorch) of some models for Sound/Audio Classification.

MFCC + TDNN (Mel Frequency Cepstral Coefficents, Time-Delay Neural Network)

## Dataset
[DCASE2020](https://zenodo.org/record/3678171)
  : ./ackit/data_utils/soundreader2020.py

20000

# Experiments Set

### CNN
- models: ./ackit/models/cnn_classifier.py
- pretrain_model: ./runs/VAE/model_epoch_12/model_epoch12.pth
- config: ./configs/autoencoder.yaml
- result: ./runs/vae_cnncls/202404181142_ptvae/
- test and plot: 

run
```commandline
python trainer_ConvEncoder.py
```
