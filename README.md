[English](README.md)|[简体中文](README_cn.md)
# Audio Classification
This repository contains the implementation (in PyTorch) of some models for Sound/Audio Classification.

MFCC + TDNN (Mel Frequency Cepstral Coefficents, Time-Delay Neural Network)

## Dataset
[DCASE2020](https://zenodo.org/record/3678171)
  : ./ackit/data_utils/soundreader2020.py

This dataset contains 20000 audio pieces, each lasting 10 seconds, and a total of 7 categories of machines, totaling 23 different machines.

# Experiments Set

## CNN
- models: ./ackit/models/cnn_classifier.py
- pretrain_model: ./runs/VAE/model_epoch_12/model_epoch12.pth
- config: ./configs/autoencoder.yaml
- result: ./runs/vae_cnncls/202404181142_ptvae/
- test and plot: ./ackit/utils/plotter.py plot_TSNE()

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

