#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/19 11:46
# @Author  : dieuroi
# @Site    : 
# @File    : data_generation.py
# @Software: PyCharm
import os
import json
import torch
import numpy as np
import pickle
import random
from tqdm import tqdm

from config import *
from AesDataset import AesDataset

# json file stored true id, while u_id, m_id here are index id

def one_hot_converting(one_dic, src_id):
    emd_size = len(list(one_dic.keys()))
    idx = torch.zeros(1, emd_size).long()
    # idx[0, int(one_dic[src_id])] = 1
    idx[0, int(src_id)] = 1
    return idx


def turn_idx_to_one_hot(src_id, voc_size=10000):
    idx = torch.zeros(1, voc_size).long()
    idx[0, int(src_id)] = 1
    return idx


def generate(master_path='./ml', dataset_path='data'):
    if not os.path.exists("{}/warm_state/".format(master_path)):
        for state in states:
            os.mkdir("{}/{}/".format(master_path, state))
    if not os.path.exists("{}/log/".format(master_path)):
        os.mkdir("{}/log/".format(master_path))

    aesdata = AesDataset()

    # hashmap for item information
    if not os.path.exists("{}/m_item_one_hot_dict.pkl".format(master_path)):
        item_dict = {}
        print('------- item one hot converting start -------')
        for i in aesdata.item_input:
            print('convert item {}'.format(i))
            i_info = one_hot_converting(aesdata.item_dic, i)
            item_dict[i] = i_info
        pickle.dump(item_dict, open("{}/m_item_one_hot_dict.pkl".format(master_path), "wb"))
        print('------- item one hot converting done -------')
    else:
        print('------- load item one hot converting -------')
        item_dict = pickle.load(open("{}/m_item_one_hot_dict.pkl".format(master_path), "rb"))
    # hashmap for user profile
    if not os.path.exists("{}/m_user_one_hot_dict.pkl".format(master_path)):
        user_dict = {}
        print('------- user one hot converting start -------')
        for u in aesdata.user_input:
            print('convert user {}'.format(u))
            u_info = one_hot_converting(aesdata.user_dic, u)
            user_dict[u] = u_info
        pickle.dump(user_dict, open("{}/m_user_one_hot_dict.pkl".format(master_path), "wb"))
        print('------- user one hot converting done -------')
    else:
        print('------- load user one hot converting -------')
        user_dict = pickle.load(open("{}/m_user_one_hot_dict.pkl".format(master_path), "rb"))

    for state in states:
        idx = 0
        if not os.path.exists("{}/{}/{}".format(master_path, "log", state)):
            os.mkdir("{}/{}/{}".format(master_path, "log", state))
        with open("{}/{}.json".format(dataset_path, state), encoding="utf-8") as f:
            dataset = json.loads(f.read())
        with open("{}/{}_y.json".format(dataset_path, state), encoding="utf-8") as f:
            dataset_y = json.loads(f.read())

        for _, user_id in tqdm(enumerate(dataset.keys())):
            if os.path.exists("{}/log/{}/query_x_{}_u_m_ids.txt".format(master_path, state, idx)):
                idx += 1
                continue

            u_id = int(user_id)
            seen_img_len = len(dataset[str(u_id)])
            indices = list(range(seen_img_len))

            random.shuffle(indices)
            tmp_x = np.array(dataset[str(u_id)])
            # tmp_y = np.array(dataset_y[str(u_id)])
            '''
            tmp_y = dataset_y[str(u_id)]
            support_query_y = list()
            for words in tmp_y:
                try:
                    support_query_y.append(np.eye(10000)[words])
                except:
                    print(words)
            support_query_y = np.array(support_query_y)
            '''
            support_query_y = np.array(dataset_y[str(u_id)])
            print(support_query_y.shape)

            support_x_app = None
            for m_id in tmp_x[indices[:-10]]:
                m_id = int(m_id)
                feature_tensor = torch.Tensor(aesdata.features[m_id]).unsqueeze(0)
                tmp_x_converted = torch.cat((item_dict[m_id].float(), user_dict[u_id].float(), feature_tensor), 1)
                try:
                    support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
                except:
                    support_x_app = tmp_x_converted

            query_x_app = None
            for m_id in tmp_x[indices[-10:]]:
                m_id = int(m_id)
                u_id = int(user_id)
                feature_tensor = torch.Tensor(aesdata.features[m_id]).unsqueeze(0)
                tmp_x_converted = torch.cat((item_dict[m_id].float(), user_dict[u_id].float(), feature_tensor), 1)
                try:
                    query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
                except:
                    query_x_app = tmp_x_converted
            support_y_app = torch.LongTensor(support_query_y[indices[:-10]])
            query_y_app = torch.LongTensor(support_query_y[indices[-10:]])

            pickle.dump(support_x_app, open("{}/{}/supp_x_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(support_y_app, open("{}/{}/supp_y_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(query_x_app, open("{}/{}/query_x_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(master_path, state, idx), "wb"))
            with open("{}/log/{}/supp_x_{}_u_m_ids.txt".format(master_path, state, idx), "w") as f:
                for m_id in tmp_x[indices[:-10]]:
                    f.write("{}\t{}\n".format(u_id, m_id))
            with open("{}/log/{}/query_x_{}_u_m_ids.txt".format(master_path, state, idx), "w") as f:
                for m_id in tmp_x[indices[-10:]]:
                    f.write("{}\t{}\n".format(u_id, m_id))
            idx += 1
