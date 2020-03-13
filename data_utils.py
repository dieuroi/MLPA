#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/18 10:07
# @Author  : dieuroi
# @Site    : 
# @File    : data_utils.py
# @Software: PyCharm
import os
# import h5py
import json
import pickle
import codecs
from collections import Counter, defaultdict
# from nltk import word_tokenize


def exchange_pos(src_file='iur.txt', dst_file='uir.txt'):
    user_dic = defaultdict(list)
    file_object = codecs.open(src_file, mode='r', encoding='utf-8')
    lines = file_object.readlines()
    lines = [line for line in lines if len(line.strip().split('|')) > 2]
    words = [line.strip().split('|')[-1] for line in lines]
    items = [line.strip().split('|')[0] for line in lines]
    users = [line.strip().split('|')[1] for line in lines]
    new_lines = list()
    for (u, i) in zip(users, items):
        user_dic[u].append(i)
    print('user numbers:', len(list(user_dic.keys())))
    # [12, 102]: 4867
    for i, line in enumerate(words):
        if 12 <= len(user_dic[users[i]]) <= 102:
            new_line = [str(users[i]), str(items[i]), line]
            new_lines.append(new_line)
    new_lines = sorted(new_lines)
    new_lines = ['|'.join(l) for l in new_lines]
    new_lines = [l+'\r\n' for l in new_lines]
    if not os.path.exists(dst_file):
        with open(dst_file, 'w') as f:
            f.writelines(new_lines)


# each line in iur file: item|user|review
def check_data(data_file='iur.txt', feat_file='feats.h5'):
    with open(data_file) as f:
        lines = f.readlines()
        lines = [line.split('|')[0] for line in lines]

    f_feat = h5py.File(feat_file,'r')
    items = []
    for key in f_feat.keys():
        # print(type(f_feat[key][:]))
        items.append(key.split('.')[0])

    impl_set = set(lines)
    item_set = set(items)
    k_set = impl_set - item_set
    print(k_set)


def gen_udata(data_file='uir.txt'):
    file_object = codecs.open(data_file, mode='r', encoding='utf-8')
    lines = file_object.readlines()
    lines = [line for line in lines if len(line.strip().split('|')) > 2]
    words = [line.strip().split('|')[-1] for line in lines]
    items = [line.strip().split('|')[1] for line in lines]
    users = [line.strip().split('|')[0] for line in lines]
    c_inputs = Counter()
    vocab_size = 10000
    word2idx = dict()
    idx2word = dict()
    itemdic = dict()
    userdic = dict()
    # reviews = list()
    new_lines = list()

    for i, imgid in enumerate(list(set(items))):
        itemdic[imgid] = i

    for i, userid in enumerate(list(set(users))):
        userdic[userid] = i

    for i, line in enumerate(words):
        tokens = word_tokenize(line)
        c_inputs.update(tokens)
        # reviewd = dict()
        # reviewd['reviewerID'] = userdic[users[i]]
        # reviewd['asin'] = itemdic[items[i]]
        # reviewd['rating'] = 1
        # reviewd['reviewText_tok'] = ' '.join(tokens)
        # new_line = '{}\t{}\t{}\r\n'.format(reviewd['reviewerID'], reviewd['asin'], reviewd['reviewText_tok'])
        # tokenized stored indexed items and users
        new_line = [str(userdic[users[i]]), str(itemdic[items[i]]), ' '.join(tokens)]
        new_lines.append(new_line)
    new_lines = sorted(new_lines)
    new_lines = ['|'.join(l) for l in new_lines]
    new_lines = [l+'\r\n' for l in new_lines]

    vocab_list = c_inputs.most_common(vocab_size)
    print(vocab_list)

    for i, tuplee in enumerate(vocab_list):
        print(tuplee)
        word, _ = tuplee
        word2idx[word] = i + 4
        idx2word[i + 4] = word

    if not os.path.exists('vocab.pickle'):
        with open('vocab.pickle', 'wb') as data_f:
            pickle.dump((word2idx, idx2word), data_f)

    # key: str, value: int
    if not os.path.exists('ui.pickle'):
        with open('ui.pickle', 'wb') as data_ui:
            pickle.dump((userdic, itemdic), data_ui)

    if not os.path.exists('uir-tokenized.txt'):
        with open('uir-tokenized.txt', 'w') as data_uir:
            data_uir.writelines(new_lines)


def statistic(x_file='data/warm_state.json', y_file='data/warm_state_y.json'):
    count = 0
    with open(x_file, encoding="utf-8") as f:
        dataset = json.loads(f.read())
    with open(y_file, encoding="utf-8") as f:
        dataset_y = json.loads(f.read())
    assert len(list(dataset.keys())) == len(list(dataset_y.keys())), 'Dataset is not equal!'
    for key in dataset.keys():
        if len(dataset[key]) >= 12:
            count += 1
    print('count:', count)


if '__main__' == __name__:
    # exchange_pos()
    statistic()
