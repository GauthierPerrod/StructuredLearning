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

In this script, we train the model WITH the attention layer
"""


#=========================================================================================================
#=========================================================================================================
#================================ 0. MODULE



from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence

from model import Encoder, DecoderWithAttention
from datasets import CaptionDataset
from nltk.translate.bleu_score import corpus_bleu



#=========================================================================================================
#=========================================================================================================
#================================ 1. HYPERPARAMETERS









#=========================================================================================================
#=========================================================================================================
#================================ 2. DEFINING ARCHITECTURE









#=========================================================================================================
#=========================================================================================================
#================================ 3. TRAINING