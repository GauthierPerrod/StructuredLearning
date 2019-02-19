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

In this script, we define:
    - the encoder (ResNet model pretrained on ImageNet)
    - the decoder (LSTM + (optional) attention layer)

Note: we will not train the encoder (maybe we'll fine-tune it)
"""



#=========================================================================================================
#================================ 0. MODULE


import torch
from torch import nn
import torchvision


#=========================================================================================================
#================================ 1. ENCODER


class Encoder(nn.Module):
    """
    Pretrained feature extractor for images and return them as
    same size encoded tensors
    """
    def __init__(self, output_size=16):
        super(Encoder, self).__init__()

        resnet = torchvision.models.resnet101(pretrained=True)

        # Remove linear and pool layers to have a fully convolutional network
        fully_convolutional_modules = list(resnet.children())[:-2]
        self.feature_extractor = nn.Sequential( *fully_convolutional_modules )

        # Freeze all parameters of feature extractor
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((output_size, output_size))


    def forward(self, images):
        """
        Forward propagation.
        
        Arguments:
        ----------
        images : tensor of shape (batch_size, 3, w, h)

        Return:
        -------
        out : tensor of shape (batch_size, output_size, output_size, 2048)
        """
        out = self.feature_extractor(images)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)
        return out


#=========================================================================================================
#================================ 2. DECODER





#=========================================================================================================
#================================ 3. ATTENTION


class Attention(nn.Module):
    """
    Attention Network, used to help the decoder to focus on the most informative
    part of the image at each step.
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        Arguments:
        ----------
        encoder_dim: feature size of encoded images
        decoder_dim: size of decoder's RNN
        attention_dim: size of the attention network
        """
        super(Attention, self).__init__()

        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights


    def forward(self, encoder_out, decoder_hidden):
        """
        Arguments:
        ----------
        encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        
        Return: 
        -------
        attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha