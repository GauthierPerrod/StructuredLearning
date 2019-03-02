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
from dataset import CaptionDataset
from nltk.translate.bleu_score import corpus_bleu

from datetime import datetime
from dateutil.relativedelta import relativedelta
def diff(t_a, t_b):
    t_diff = relativedelta(t_a, t_b)
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)



#=========================================================================================================
#=========================================================================================================
#================================ 1. HYPERPARAMETERS



# Data
data_folder = '/home/hugoperrin/Bureau/Datasets/Coco/data/'
MIN_WORD_FREQ = 5
N_CAPTIONS = 5
base_filename = 'COCO_' + str(N_CAPTIONS) + '_cap_per_img_' + str(MIN_WORD_FREQ) + '_min_word_freq'

# Model
EMBBEDING_DIM = 512
ATTENTION_DIM = 512  
DECODER_DIM = 512
DROPOUT = 0.5
DEVICE = 'cuda:0'  
cudnn.benchmark = True

# Training
EPOCHS = 50 
BATCH_SIZE = 64
LEARNING_RATE = 5e-4

GRAD_CLIP = 5.    
ALPHA = 1.              # regularization parameter for 'doubly stochastic attention', as in the paper
BB = 0.                 # BLEU-4 score right now
DISPLAY_STEP = 100



#=========================================================================================================
#=========================================================================================================
#================================ 2. DEFINING ARCHITECTURE



# Read word map
word_map_file = os.path.join(data_folder, 'WORDMAP_' + base_filename + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)


# Networks
decoder = DecoderWithAttention().to(device)
encoder = Encoder().to(device)

# Loss function
criterion = nn.CrossEntropyLoss().to(device)

# Data loader
train_loader = torch.utils.data.DataLoader( 
        CaptionDataset(data_folder, data_name, 'TRAIN'), 
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL'),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

# Optimizer
optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)



#=========================================================================================================
#=========================================================================================================
#================================ 3. TRAINING


for epoch in EPOCHS:
    decoder.train()
    encoder.train()

    time = datetime.now()

    for i, (image, caption, length) in enumerate(train_loader):

        # Batch data
        image = image.to(device)
        caption = caption.to(device)
        length = length.to(device)

        # Forward
        encoded_image = encoder(image)
        scores, caption_sorted, decode_lengths, alphas, sort_idx = decoder(encoded_image, caption, length)











    print('\nEpoch time: ', diff(datetime.now(), time))
