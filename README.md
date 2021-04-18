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

I regularize the label by dividing label with scale [100, 1000, 1000]

Revise the effect of label's regularization:

1. intro2dl_mid.py

In this version I mutiply the regularization scale to the result.

Though the result seems normal and some predictions close to true label, the loss is quite high.

