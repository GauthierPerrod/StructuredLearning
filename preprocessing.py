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

In this script, we preprocess the image, pass them through the encoder and save the output.
We do this save time during training since there is only the decoder and the attention to train!
"""



#=========================================================================================================
#=========================================================================================================
#================================ 0. MODULE



import os
import numpy as np

import h5py
import json

import torch
import torchvision.transforms as transforms

from skimage.transform import resize
from imageio import imread

from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import warnings
warnings.filterwarnings('ignore')



#=========================================================================================================
#=========================================================================================================
#================================ 1. CREATE DATA



CAPTION_PATH = '/home/hugoperrin/Bureau/Datasets/Coco/caption_datasets/dataset_coco.json'
IMAGE_PATH = '/home/hugoperrin/Bureau/Datasets/Coco/'
OUTPUT_PATH = '/home/hugoperrin/Bureau/Datasets/Coco/data/'

## Results in approximately 5Go
TRAIN_SIZE = 10000
TEST_SIZE = 1000
VALID_SIZE = 1000

MAX_LEN = 30
MIN_WORD_FREQ = 5
N_CAPTIONS = 5


## CREATE ROOT NAME
base_filename = 'COCO_' + str(N_CAPTIONS) + '_cap_per_img_' + str(MIN_WORD_FREQ) + '_min_word_freq'


def create_data():

    ## READ IMAGE PATH AND CAPTIONS
    with open(CAPTION_PATH, 'r') as j:
        data = json.load(j)

    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    n_train = 0
    n_test = 0
    n_valid = 0


    print('Reading images path and captions')
    for img in tqdm(data['images']):

        path = os.path.join(IMAGE_PATH, img['filepath'], img['filename'])

        if not os.path.isfile(path):
            continue
            
        if img['split'] in {'train', 'restval'}:
            if n_train >= TRAIN_SIZE:
                continue
            else:
                n_train += 1
        elif img['split'] in {'val'}:
            if n_valid >= VALID_SIZE:
                continue
            else:
                n_valid += 1
        elif img['split'] in {'test'}:
            if n_test >= TEST_SIZE:
                continue
            else:
                n_test += 1
            
        
        captions = []
        for c in img['sentences']:
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= MAX_LEN:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(IMAGE_PATH, img['filepath'], img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    ## CREATE WORD MAP
    print('Creating word map', end='...')
    words = [w for w in word_freq.keys() if word_freq[w] > MIN_WORD_FREQ]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    with open(os.path.join(OUTPUT_PATH, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)
    print('done')

    ## SAMPLE CAPTIONS FOR EACH IMAGE, GET ENCODING OF IMAGE AND SAVE IT AS HDF5 FILE
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(OUTPUT_PATH, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:

            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = N_CAPTIONS

            # Create dataset inside HDF5 file to store encoded images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='float16')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < N_CAPTIONS:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(N_CAPTIONS - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=N_CAPTIONS)

                # Read images
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = resize(img, (256, 256))
                img = img.transpose(2, 0, 1)

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (MAX_LEN - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(OUTPUT_PATH, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(OUTPUT_PATH, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)



#=========================================================================================================
#=========================================================================================================
#================================ 2. LOAD EMBEDDING



def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    Argument:
    ---------
    embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)



def load_embeddings(emb_file, folder):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    Argument:
    ---------
    emb_file: file containing embeddings (stored in GloVe format)
    folder: folder containing the word map
    
    Return:
    -------
    embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    # Open vocab
    word_map_file = os.path.join(folder, 'WORDMAP_' + base_filename + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim



def print_closest(word_map, word, n=10):
    """
    Sanity check for the pre-trained embedding:
        >> WORKING
    """
    word_idx = {i: v for v, i in word_map.items()}
    idx = word_map[word]
    vector = embedding[idx]
    distances = torch.pairwise_distance(embedding, vector)
    _, idx = distances.sort(descending=False)
    for k in range(n):
        print(word_idx[idx[k].item()])



#=========================================================================================================
#=========================================================================================================
#================================ 3. MAIN



if __name__ == "__main__":
    create_data()