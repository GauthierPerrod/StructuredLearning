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

device = 'cuda:0'



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
        images : tensor of shape (batch_size, 3, h, w)

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



class DecoderWithAttention(nn.Module):
    """
    Decoder: 

    RNN taking the encoded images as input to predict corresponding sentences
    """
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.3):
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer

        self.dropout = nn.Dropout(p=self.dropout)

        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell

        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary

        self.init_weights()  # initialize some layers with the uniform distribution


    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, 
        for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)


    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings. If so, 
        the parameters are freezed.
        
        Arguments:
        ----------
        embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

        for p in self.embedding.parameters():
            p.requires_grad = False


    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        Arguments:
        ----------
        encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        
        Return:
        -------
        hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c


    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation

        Arguments:
        ----------
        encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        
        Return:
        -------
        scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):

            batch_size_t = sum([l > t for l in decode_lengths])

            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])

            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)

            attention_weighted_encoding = gate * attention_weighted_encoding

            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind



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
        Forward propagation

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