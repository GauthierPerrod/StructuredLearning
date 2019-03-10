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


import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import json
import os 

from model import Encoder, Decoder
from preprocessing import load_embeddings

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
from skimage.transform import resize
from imageio import imread

import argparse


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


#=========================================================================================================
#=========================================================================================================
#================================ 2. BEAM SEARCH




def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=5, with_attention=False):
    """
    Reads an image and captions it with beam search.

    Arguments:
    ----------
    encoder: encoder model
    decoder: decoder model
    image_path: path to image
    word_map: word map
    beam_size: number of sequences to consider at each decode-step
    
    Returns:
    --------
    caption
    weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    decoder.eval()
    encoder.eval()

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = resize(img, (256, 256))
    img = img.transpose(2, 0, 1)

    # Encode
    image = torch.FloatTensor(img).unsqueeze(0).to(DEVICE)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(DEVICE)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(DEVICE)  # (k, 1, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(DEVICE)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = []
    complete_seqs_alpha = []
    complete_seqs_scores = []

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    if with_attention:
        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            try:

                embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

                awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

                alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

                gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
                awe = gate * awe

                h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

                scores = decoder.classifier(h)  # (s, vocab_size)
                scores = scores.squeeze()
                scores = F.log_softmax(scores, dim=1)

                # Add
                scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words / vocab_size  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)

                # Add new words to sequences, alphas
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
                seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                                       dim=1)  # (s, step+1, enc_image_size, enc_image_size)

                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly

                # Proceed with incomplete sequences
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
                seqs_alpha = seqs_alpha[incomplete_inds]
                h = h[prev_word_inds[incomplete_inds]]
                c = c[prev_word_inds[incomplete_inds]]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

                # Break if things have been going on too long
                if step > 50:
                    break
                step += 1

            except:
                break

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]

        return seq, alphas

    else:
        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        hidden = (h.unsqueeze(0), c.unsqueeze(0))

        while True:

            try:
                embeddings = decoder.embedding(k_prev_words)  # (s, embed_dim)

                out, hidden = decoder.decoder(embeddings, hidden)  # (s, decoder_dim)

                scores = decoder.classifier( out )  # (s, vocab_size)
                scores = scores.squeeze()
                scores = F.log_softmax(scores, dim=1)

                # Add
                scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words / vocab_size  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)

                # Add new words to sequences, alphas
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly

                # Proceed with incomplete sequences
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
                h = h[prev_word_inds[incomplete_inds]]
                c = c[prev_word_inds[incomplete_inds]]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

                # Break if things have been going on too long
                if step > 50:
                    break
                step += 1

            except:
                break

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        return seq, i



#=========================================================================================================
#=========================================================================================================
#================================ 3. MAIN


# Same as in training
ATTENTION = False

DATA_FOLDER = '../datasets/Coco/data/'
MIN_WORD_FREQ = 5
N_CAPTIONS = 5

base_filename = 'COCO_' + str(N_CAPTIONS) + '_cap_per_img_' + str(MIN_WORD_FREQ) + '_min_word_freq'
embedding_file = '../datasets/glove.6B.200d.txt'

ENCODER_DIM = 2048      # ResNet
ATTENTION_DIM = 512
EMBBEDING_DIM = 200
DECODER_DIM = 512
DROPOUT = 0.3

DEVICE = 'cuda:0'


LAST_EPOCH = 26


PATH_IMAGES = ['../datasets/Coco/train2014/COCO_train2014_000000000034.jpg',
               '../datasets/Coco/train2014/COCO_train2014_000000000078.jpg',
               '../datasets/Coco/train2014/COCO_train2014_000000000081.jpg',
               '../datasets/Coco/train2014/COCO_train2014_000000000110.jpg',
               '../datasets/Coco/train2014/COCO_train2014_000000000194.jpg',
               '../datasets/Coco/train2014/COCO_train2014_000000000263.jpg',
               '../datasets/Coco/train2014/COCO_train2014_000000000394.jpg',
               '../datasets/Coco/train2014/COCO_train2014_000000000404.jpg',
               '../datasets/Coco/train2014/COCO_train2014_000000000431.jpg',
               '../datasets/Coco/train2014/COCO_train2014_000000000471.jpg',
               '../datasets/Coco/train2014/COCO_train2014_000000000510.jpg',
               '../datasets/Coco/train2014/COCO_train2014_000000000656.jpg',
               '../datasets/Coco/train2014/COCO_train2014_000000000813.jpg',
               '../datasets/Coco/train2014/COCO_train2014_000000000828.jpg',
               '../datasets/Coco/train2014/COCO_train2014_000000001098.jpg']

k = 5

if __name__ == '__main__':

    # Read word map
    print('\nLoading word map', end='...')
    word_map_file = os.path.join(DATA_FOLDER, 'WORDMAP_' + base_filename + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    vocab_size = len(word_map)
    print('done')

    # Load networks
    print('Loading networks', end='...')
    if ATTENTION:
        decoder = DecoderWithAttention(ATTENTION_DIM, EMBBEDING_DIM, DECODER_DIM, 
                                       vocab_size, ENCODER_DIM, DROPOUT)
    else:   
        decoder = Decoder(EMBBEDING_DIM, DECODER_DIM, vocab_size, ENCODER_DIM, DROPOUT)
    print('done')

    print('Loading last weights', end='...')
    decoder.load_state_dict(torch.load('../models/image_captioning_{}.model'.format(LAST_EPOCH)))
    encoder = Encoder(output_size=12)                                                               ## CAREFUL                                                   
    print('done')

    # Load embedding
    print('Load embeddings', end='...')
    embedding, _ = load_embeddings(embedding_file, DATA_FOLDER)
    decoder.load_pretrained_embeddings(embedding)
    print('done\n')

    encoder = encoder.to(DEVICE)
    decoder = decoder.to(DEVICE)

    # Beam search
    for image in  PATH_IMAGES:
        seq, _ = caption_image_beam_search(encoder, decoder, image, word_map, k, ATTENTION)
        idx_to_word = {v: k for k, v in word_map.items()}
        tokens = [idx_to_word[i] for i in seq]
        predicted_description = ' '.join(tokens[1:-1])
        print(predicted_description)