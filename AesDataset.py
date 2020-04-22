#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/18 9:35
# @Author  : dieuroi
# @Site    : 
# @File    : AesDataset.py
# @Software: PyCharm
import os
import re
import h5py
import json
import pickle
import itertools
import unicodedata
import scipy.sparse as sp
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from config import *

np.random.seed(7)


class Voc:
    def __init__(self, name):
        self.name = name

        with open('data/vocab.pickle', 'rb') as fp:
            self.word2idx, self.idx2word = pickle.load(fp)
            self.n_words = len(self.word2idx)

class BaseDataset(Dataset):
    def __init__(self):
        self.voc = Voc('aes')

    def indexesFromSentence(self, sentence):
        ids = []
        if isinstance(sentence, str):
            sentence = sentence.strip().split(' ')
        for word in sentence:
            if word.lower() in self.voc.word2idx:
                ids.append(self.voc.word2idx[word.lower()])
            else:
                ids.append(UNK_token)
        return ids

    def pad_list(self, lst):
        if len(lst) < MAX_LENGTH:
            lst.extend([0] * (MAX_LENGTH - len(lst)))
            # print(len(lst))
            # np.array can not store in json
            # return np.array(lst)
            return lst
        else:
            return lst[:MAX_LENGTH]
            # print(len(lst[:MAX_LENGTH]))
            # return np.array(lst[:MAX_LENGTH])

    def partition(self, ui_file='data/ui.pickle', data_file='data/uir-tokenized.txt'):
        warm_dic = defaultdict(list)
        u_cold_dic = defaultdict(list)
        i_cold_dic = defaultdict(list)
        ui_cold_dic = defaultdict(list)
        warm_dic_y = defaultdict(list)
        u_cold_dic_y = defaultdict(list)
        i_cold_dic_y = defaultdict(list)
        ui_cold_dic_y = defaultdict(list)
        warm_i = defaultdict(list)
        u_cold_i = defaultdict(list)
        i_cold_i = defaultdict(list)
        ui_cold_i = defaultdict(list)
        word_dic = defaultdict(lambda:defaultdict())
        with open(data_file) as fword:
            lines = fword.readlines()
            lines = [l.strip().split('|') for l in lines]
        # word_dic is a dict for word_dic[uid][iid]
        for line in lines:
            # item_word_dic = dict()
            # item_word_dic[int(line[1])] = self.pad_list(self.indexesFromSentence(line[2]))
            word_dic[int(line[0])][int(line[1])] = self.pad_list(self.indexesFromSentence(line[2]))  # word_dic['100002']['771807']

        with open(ui_file, 'rb') as data_ui:
            userdic, itemdic = pickle.load(data_ui)
        userlist = [userdic[u] for u in userdic.keys()]
        u_part = int(len(userlist) * 0.2)
        itemlist = [itemdic[i] for i in itemdic.keys()]
        i_part = int(len(itemlist) * 0.2)
        existed_u = userlist[u_part:]
        new_u = userlist[:u_part]
        existed_i = itemlist[i_part:]
        new_i = itemlist[:i_part]
        for eu in existed_u:
            for ei in existed_i:
                if eu in word_dic and ei in word_dic[eu]:
                    warm_dic[eu].append(ei)
                    warm_dic_y[eu].append(word_dic[eu][ei])
                    warm_i[ei].append(word_dic[eu][ei])
            for ni in new_i:
                if eu in word_dic and ni in word_dic[eu]:
                    i_cold_dic[eu].append(ni)
                    i_cold_dic_y[eu].append(word_dic[eu][ni])
                    i_cold_i[ni].append(word_dic[eu][ni])
        for nu in new_u:
            for ei in existed_i:
                if nu in word_dic and ei in word_dic[nu]:
                    u_cold_dic[nu].append(ei)
                    u_cold_dic_y[nu].append(word_dic[nu][ei])
                    u_cold_i[ei].append(word_dic[nu][ei])
            for ni in new_i:
                if nu in word_dic and ni in word_dic[nu]:
                    ui_cold_dic[nu].append(ni)
                    ui_cold_dic_y[nu].append(word_dic[nu][ni])
                    ui_cold_i[ni].append(word_dic[nu][ni])
        for key in list(warm_dic.keys()):
            if len(warm_dic[key]) < 12 or len(warm_dic[key]) > 102:
                del warm_dic[key]
                del warm_dic_y[key]
        for key in list(u_cold_dic.keys()):
            if len(u_cold_dic[key]) < 12 or len(u_cold_dic[key]) > 102:
                del u_cold_dic[key]
                del u_cold_dic_y[key]
        for key in list(i_cold_dic.keys()):
            if len(i_cold_dic[key]) < 12 or len(i_cold_dic[key]) > 102:
                del i_cold_dic[key]
                del i_cold_dic_y[key]
        for key in list(ui_cold_dic.keys()):
            if len(ui_cold_dic[key]) < 12 or len(ui_cold_dic[key]) > 102:
                del ui_cold_dic[key]
                del ui_cold_dic_y[key]
        print('warm:', len(list(warm_dic)))
        print('u_cold:', len(list(u_cold_dic)))
        print('i_cold:', len(list(i_cold_dic)))
        print('ui_cold:', len(list(ui_cold_dic)))
        with open('data/warm_state.json', 'w') as fw:
            json.dump(warm_dic, fw)
        with open('data/user_cold_state.json', 'w') as fu:
            json.dump(u_cold_dic, fu)
        with open('data/item_cold_state.json', 'w') as fi:
            json.dump(i_cold_dic, fi)
        with open('data/user_and_item_cold_state.json', 'w') as fui:
            json.dump(ui_cold_dic, fui)
        with open('data/warm_state_y.json', 'w') as fwy:
            json.dump(warm_dic_y, fwy)
        with open('data/user_cold_state_y.json', 'w') as fuy:
            json.dump(u_cold_dic_y, fuy)
        with open('data/item_cold_state_y.json', 'w') as fiy:
            json.dump(i_cold_dic_y, fiy)
        with open('data/user_and_item_cold_state_y.json', 'w') as fuiy:
            json.dump(ui_cold_dic_y, fuiy)
        # with open('data/warm_state_i.json', 'w') as fwii:
        #     json.dump(warm_i, fwii)
        # with open('data/user_cold_state_i.json', 'w') as fuii:
        #     json.dump(u_cold_i, fuii)
        # with open('data/item_cold_state_i.json', 'w') as fiii:
        #     json.dump(i_cold_i, fiii)
        # with open('data/user_and_item_state_i.json', 'w') as fuiii:
        #     json.dump(ui_cold_i, fuiii)


class AesDataset(BaseDataset):
    'Characterizes the dataset for PyTorch, and feeds the (user,item) pairs for training'

    def __init__(self, file_name='data/uir.txt', feat_file='data/feats.h5', ui_file='data/ui.pickle'):
        'Load the datasets from disk, and store them in appropriate structures'
        super(AesDataset, self).__init__()
        self.user_dic, self.item_dic = self.get_ui(ui_file)
        self.user_input, self.item_input, self.reviews, self.features = self.get_instances(file_name, feat_file)
        # make testing set with negative sampling
        # self.testRatings, _, _, _ = self.load_rating_file_as_list(file_name + ".test.rating")
        # self.testNegatives = self.create_negative_file(num_samples=num_negatives_test)
        # assert len(self.testRatings) == len(self.testNegatives)

    def __len__(self):
        'Denotes the total number of rating in test set'
        return len(self.user_input)

    def __getitem__(self, index):
        'Generates one sample of data'

        # get the train data
        user_id = self.user_input[index]
        item_id = self.item_input[index]
        review = torch.from_numpy(self.reviews[index]).type(torch.LongTensor)
        feature = torch.from_numpy(self.features[item_id]).type(torch.FloatTensor)

        return {'user_id': user_id,
                'item_id': item_id,
                'review': review,
                'feature': feature}

    def get_ui(self, ui_file):
        with open(ui_file, 'rb') as data_ui:
            userdic, itemdic = pickle.load(data_ui)
        return userdic, itemdic

    def get_instances(self, file_name, feat_file):
        feature_dic = dict()
        user_input = list()
        item_input = list()
        reviews = list()
        f_feat = h5py.File(feat_file, 'r')
        for key in self.item_dic.keys():
            image_name = key + '.jpg'
            feature_dic[self.item_dic[key]] = f_feat[image_name][:]
        with open(file_name, 'r') as f:
            lines = f.readlines()
            lines = [line.strip().split('|') for line in lines]
        for line in lines:
            user_input.append(self.user_dic[line[0]])
            item_input.append(self.item_dic[line[1]])
            review = self.indexesFromSentence(line[2])
            reviews.append(self.pad_list(review))

        return user_input, item_input, reviews, feature_dic


# temporally not used
def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    m = []
    for i in range(len(l)):
        m.append([])
        for j in range(len(l[i])):
            if l[i][j] == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def encode_dict_with_start_end(fn='data/warm_state_y.json'):
    with open(fn, 'r') as f:
        truthdic = json.load(f)
    for key in truthdic:
        sents = truthdic[key]
        new_sents = list()
        for sent in sents:
            sent.insert(0, SOS_token)
            if 0 not in sent:
                sent.insert(99, EOS_token)
                sent = sent[:100]
            else:
                sent.insert(sent.index(0), EOS_token)
                sent = sent[:100]
            new_sents.append(sent)
        truthdic[key] = new_sents
    with open(fn, 'w') as fw:
        json.dump(truthdic, fw)


if '__main__' == __name__:

    '''
    train_dataset = AesDataset()
    training_data_generator = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=10)
    for batch, training_dict in enumerate(training_data_generator):
        attr_input = [training_dict['user_id'], training_dict['item_id']]
        review_input = training_dict['review']
        image_input = training_dict['feature']

    dataset = BaseDataset()
    dataset.partition()
    '''
    encode_dict_with_start_end()
    encode_dict_with_start_end('data/user_cold_state_y.json')
    encode_dict_with_start_end('data/item_cold_state_y.json')
    encode_dict_with_start_end('data/user_and_item_cold_state_y.json')

