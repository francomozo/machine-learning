# Gradual unfreezing experiment

This folder contains the code to study how gradual unfreezing affects performance compared to training the whole model and a two step training strategy - converge head the whole model - on a pretrained model.

> This code is heavily hardcoded and could do some improvement.

## Data
The dataset used for pretraining is the CIFAR10 dataset, and the dataset use to finetune is the Animals10 dataset which can be found [here](https://www.kaggle.com/datasets/alessiocorrado99/animals10).