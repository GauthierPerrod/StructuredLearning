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

In this script, we train the model WITHOUT the attention layer
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

from model import Encoder, Decoder
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
embedding_file = '/home/hugoperrin/Bureau/Datasets/Glove/glove.6B.200d.txt'

# Model
ENCODER_DIM = 2048      # ResNet
EMBBEDING_DIM = 200
DECODER_DIM = 512
DROPOUT = 0.3
DEVICE = 'cuda:0'  
cudnn.benchmark = True

# Training
EPOCH = 50 
BATCH_SIZE = 32
LEARNING_RATE = 5e-4

GRAD_CLIP = 5.    
BB = 0.                 # BLEU-4 score right now
DISPLAY_STEP = 100



#=========================================================================================================
#=========================================================================================================
#================================ 2. DEFINING ARCHITECTURE



# Read word map
word_map_file = os.path.join(data_folder, 'WORDMAP_' + base_filename + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
vocab_size = len(word_map)

# Networks
decoder = Decoder(EMBBEDING_DIM, DECODER_DIM, vocab_size, ENCODER_DIM, DROPOUT).to(DEVICE)
encoder = Encoder(output_size=10).to(DEVICE)

# Embedding
embedding, _ = load_embeddings(embedding_file, data_folder)
decoder.load_pretrained_embeddings(embedding)

# Loss function
criterion = nn.CrossEntropyLoss().to(DEVICE)

# Data loader
train_loader = torch.utils.data.DataLoader( 
        CaptionDataset(data_folder, data_name, 'TRAIN'), 
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

valid_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL'),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

# Optimizer
optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)

# Parameters check
model_parameters = filter(lambda p: p.requires_grad, decoder.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('>> {} parameters\n'.format(params))



#=========================================================================================================
#=========================================================================================================
#================================ 3. TRAINING


for epoch in EPOCHS:
    decoder.train()
    encoder.train()
    epoch_loss = 0.

    time = datetime.now()

    for i, (image, caption, length) in enumerate(train_loader):

        # Batch data
        image = image.to(DEVICE)
        caption = caption.to(DEVICE)
        length = length.to(DEVICE)

        # Forward
        encoded_image = encoder(image)
        scores, caption_sorted, decode_lengths, sort_idx = decoder(encoded_image, caption, length)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caption_sorted[:, 1:]

        # Padding sequences
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Compute loss
        loss = criterion(scores, targets)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Clipping to avoid exploding gradient
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.clamp_(-GRAD_CLIP, GRAD_CLIP)

        # Monitoring performance
        epoch_loss += loss.data.item()

        if i % DISPLAY_STEP == DISPLAY_STEP-1:
            print('training loss: %.3f' % (epoch_loss / i))

    print('\nEpoch time: ', diff(datetime.now(), time))

    # Computing validation BLEU score
    decoder.eval()
    encoder.eval()

    for i, (image, caption, length, allcaptions) in enumerate(valid_loader):

        # Batch data
        image = image.to(DEVICE)
        caption = caption.to(DEVICE)
        length = length.to(DEVICE)

        # Forward
        encoded_image = encoder(image)
        scores, caption_sorted, decode_lengths, sort_idx = decoder(encoded_image, caption, length)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caption_sorted[:, 1:]

        # Padding sequences
        scores_unpadded = scores.clone()
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # References
        allcaptions = allcaptions[sort_idx]                          # because images were sorted in the decoder
        for j in range(allcaptions.shape[0]):
            img_captions = allcaptions[j].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                    img_caps))                                       # remove <start> and pads
            references.append(img_captions)

        # Hypotheses
        _, prediction = torch.max(scores_unpadded, dim=2)
        prediction = prediction.tolist()
        temp_preds = list()
        for j, p in enumerate(prediction):
            temp_preds.append(prediction[j][:decode_lengths[j]])     # remove pads
        prediction = temp_preds
        hypotheses.extend(prediction)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    # Monitoring performance
    print('Epoch: %2d, validation bleu-4 score: %.2f\n' % (bleu4))
    torch.save(decoder.state_dict(), "../models/image_captioning_{}.model".format(epoch))