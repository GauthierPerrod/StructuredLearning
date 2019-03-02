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

In this script, we define a class to:
    - Loop over captions and assign correct image
    - Avoid loading all images in batches
"""



#=========================================================================================================
#=========================================================================================================
#================================ 0. MODULE



import torch
from torch.utils.data import Dataset

import h5py
import json
import os



#=========================================================================================================
#=========================================================================================================
#================================ 1. CAPTION DATASET


MIN_WORD_FREQ = 5
N_CAPTIONS = 5


## CREATE ROOT NAME
base_filename = 'COCO_' + str(N_CAPTIONS) + '_cap_per_img_' + str(MIN_WORD_FREQ) + '_min_word_freq'



class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, data_folder, split, transform=None):
        """
        Arguments:
        ----------
        data_folder: folder where data files are stored
        split: one of 'TRAIN', 'VAL', or 'TEST'
        transform: image transform pipeline
        """
        self.split = split

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + base_filename + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + base_filename + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + base_filename + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)


    def __getitem__(self, i):
        # The Nth caption corresponds to the (N // captions_per_image)th image

        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)

        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])

            return img, caption, caplen, all_captions


    def __len__(self):
        return self.dataset_size