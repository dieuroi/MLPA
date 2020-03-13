#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/17 17:49
# @Author  : dieuroi
# @Site    : 
# @File    : Me.py
# @Software: PyCharm
import math
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from copy import deepcopy
from collections import OrderedDict
from config import *
from torch.nn.utils.rnn import pack_padded_sequence



class Encoder(nn.Module):

    def __init__(self, config):
        """
        Simple Feedforward network with Embeddings for users and items
        """
        super().__init__()
        layers = config['layers']
        n_users = config['n_users']
        n_items = config['n_items']
        feature_dim = config['feature_dim']
        assert (layers[0] % 2 == 0), "layers[0] must be an even number"
        self.__alias__ = "MLP {}".format(layers)

        # user and item embedding layers
        embedding_dim = int(layers[0]/2)
        # out of user nn.Embedding is [n, n_users, embedding_dim]
        # self.user_embedding = torch.nn.Embedding(n_users, embedding_dim)
        # self.item_embedding = torch.nn.Embedding(n_items, embedding_dim)
        self.user_embedding = torch.nn.Linear(n_users, embedding_dim)
        self.item_embedding = torch.nn.Linear(n_items, embedding_dim)
        self.feat_embedding = torch.nn.Linear(feature_dim, embedding_dim * 2)
        self.fc_in = embedding_dim * 4
        self.fc_out = config['encoder_dim']
        self.linear_out = torch.nn.Linear(self.fc_in, self.fc_out)

    def forward(self, x):
        # long() is used for input to nn.Embedding
        # items = x[:, :56242].long()
        # users = x[:, 56242:61109].long()
        items = x[:, :56242]
        users = x[:, 56242:61109]
        features = x[:, 61109:]
        user_embedding = self.user_embedding(users)
        item_embedding = self.item_embedding(items)
        feat_embedding = self.feat_embedding(features)
        # concatenate user and item embeddings to form input
        x = torch.cat([user_embedding, item_embedding, feat_embedding], 1)
        x = self.linear_out(x)
        return x


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, config):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = config['encoder_dim']
        self.attention_dim = config['attention_dim']
        self.embed_dim = config['embed_dim']
        self.decoder_dim = config['decoder_dim']
        self.vocab_size = config['vocab_size']
        self.dropout = config['dropout']

        self.attention = Attention(self.encoder_dim, self.decoder_dim, self.attention_dim)  # attention network

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(self.embed_dim + self.encoder_dim, self.decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(self.encoder_dim, self.decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(self.encoder_dim, self.decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, y):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param y: encoded captions, a tensor of dimension (batch_size, max_caption_length)  [containing caption lengths, a tensor of dimension (batch_size, 1)]
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        if config['Debug']:
            print('input sentences:', y.shape)

        caption_lengths = [int(cap.float().cpu().norm(0).numpy()) for cap in y]
        caption_lengths = torch.from_numpy(np.array(caption_lengths))

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = y[sort_ind]
        if config['Debug']:
            print('encoded_captions:', encoded_captions)

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).cuda()
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).cuda()

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


class Encoder_Decoder(nn.Module):
    def __init__(self, config):
        super(Encoder_Decoder, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = DecoderWithAttention(config)

    def forward(self, x, y):
        imgs = self.encoder(x)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(imgs, y)
        return scores, caps_sorted, decode_lengths, alphas, sort_ind


class MLPA(nn.Module):
    def __init__(self, config):
        super(MLPA, self).__init__()
        self.use_cuda = config['use_cuda']
        self.model = Encoder_Decoder(config)
        self.local_lr = config['local_lr']
        self.criterion = nn.CrossEntropyLoss()
        self.store_parameters()
        self.meta_optim = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        self.local_update_target_weight_name = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'linear_out.weight', 'linear_out.bias']

    def store_parameters(self):
        self.keep_weight = deepcopy(self.model.state_dict())
        self.weight_name = list(self.keep_weight.keys())
        self.weight_len = len(self.keep_weight)
        self.fast_weights = OrderedDict()

    def cal_loss(self, scores, caps_sorted, decode_lengths):
        # we have to get caps after <start>
        print('scores:', scores.shape)
        print('targets:', caps_sorted.shape)
        targets = caps_sorted[:, 1:]
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        print('scores:', scores.shape)
        print('targets:', targets.shape)
        input()
        loss = self.criterion(scores, targets)
        return loss

    def forward(self, support_set_x, support_set_y, query_set_x, query_set_y, num_local_update):
        for idx in range(num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.model.state_dict().values())
            scores, caps_sorted, decode_lengths, alphas, sort_ind = self.model(support_set_x, support_set_y)
            loss = self.cal_loss(scores, caps_sorted, decode_lengths)
            self.model.zero_grad()
            grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            # local update
            for i in range(self.weight_len):
                if self.weight_name[i] in self.local_update_target_weight_name:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - self.local_lr * grad[i]
                else:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i]
        self.model.load_state_dict(self.fast_weights)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.model(query_set_x, query_set_y)
        self.model.load_state_dict(self.keep_weight)
        loss = self.cal_loss(scores, caps_sorted, decode_lengths)
        return loss

    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys, num_local_update):
        batch_sz = len(support_set_xs)
        losses_q = []
        if self.use_cuda:
            for i in range(batch_sz):
                support_set_xs[i] = support_set_xs[i].cuda()
                support_set_ys[i] = support_set_ys[i].cuda()
                query_set_xs[i] = query_set_xs[i].cuda()
                query_set_ys[i] = query_set_ys[i].cuda()
        for i in range(batch_sz):
            loss_q = self.forward(support_set_xs[i], support_set_ys[i], query_set_xs[i], query_set_ys[i], num_local_update)
            losses_q.append(loss_q)
        losses_q = torch.stack(losses_q).mean(0)
        self.meta_optim.zero_grad()
        losses_q.backward()
        self.meta_optim.step()
        self.store_parameters()
        return

    def get_weight_avg_norm(self, support_set_x, support_set_y, num_local_update):
        tmp = 0.
        if self.cuda():
            support_set_x = support_set_x.cuda()
            support_set_y = support_set_y.cuda()
        for idx in range(num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.model.state_dict().values())
            scores, caps_sorted, decode_lengths, alphas, sort_ind = self.model(support_set_x, support_set_y)
            loss = self.cal_loss(scores, caps_sorted, decode_lengths)
            # unit loss
            loss /= torch.norm(loss).tolist()
            self.model.zero_grad()
            grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            for i in range(self.weight_len):
                # For averaging Forbenius norm.
                tmp += torch.norm(grad[i])
                if self.weight_name[i] in self.local_update_target_weight_name:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - self.local_lr * grad[i]
                else:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i]
        return tmp / num_local_update


def make_zeros(shape, cuda=False):
    zeros = torch.zeros(shape)
    if cuda:
        zeros = zeros.cuda()
    return zeros

def test_for_nan(x, name="No name given"):
    if torch.isnan(x).sum() > 0:
        print("{} has NAN".format(name))
        exit()

