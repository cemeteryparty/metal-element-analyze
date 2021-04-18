# Analyze metal element in Picture

## Description

Given 177 pictures, the tasks is to analyze the metal element of it.

For each picture, we need to analyze the copper content(%), gold and silver content(ppm).

We try to use simple CNN structure to complete this task.

## Note

### Data Augmentation

The original dataset is too small, so I sampling in dataset, and do some proc to img (rotate or flip).

### Label's regularization

Original value of label will cause gradient burst, so we need to regularize it.

We regularize the label by dividing label with scale [100, 1000, 1000]

Revise the effect of label's regularization:

1. rc-net_mirror-scale.py

In this version we mutiply the regularization scale to the the `frond model (fit and train on regularized label)'s` result.

Though the result seems normal and some predictions close to true label, the acc is quite low.

2. rc-net_2steps.py

In this version we add 2 dense layers after the `frond model` and fit it to true data.

2 dense layers: 2 dense layers perform well than 1 dense layer, and 3 dense layers will result in overfitting to some specific data.

The result is greater than `rc-net_mirror-scale`, acc is higher than it and the loss reduces.