#!/usr/bin/env python
# -*- coding: utf-8 -*-



"""
IMAGE CAPTIONING ON PYTORCH

Original paper:

Show and tell: A neural image caption generator, 
Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan, 2015


Reference for implementation:

https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/22_Image_Captioning.ipynb
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

In this script, we implement two methods to predict a caption given a sentence:
    - sampling
    - beam search
"""



#=========================================================================================================
#=========================================================================================================
#================================ 0. MODULE



#=========================================================================================================
#=========================================================================================================
#================================ 1. ACCURACY


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    Arguments:
    ----------
    scores: scores from the model
    targets: true labels
    k: k in top-k accuracy

    Return:
    -------
    top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum() 
    return correct_total.item() * (100.0 / batch_size)