#!/usr/bin/env python
# -*- coding: utf-8 -*-



"""
IMAGE CAPTIONING ON PYTORCH

Original paper:

Show and tell: A neural image caption generator, 
Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan, 2015


Reference for implementation:

https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/22_Image_Captioning.ipynb


In this script, we define:
    - the encoder (ResNet model pretrained on ImageNet)
    - the decoder (LSTM + (optional) attention layer)

Note: we will not train the encoder (maybe we'll fine-tune it)
"""



#=========================================================================================================
#=========================================================================================================
#================================ 0. MODULE