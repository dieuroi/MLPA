#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/1 18:45
# @Author  : dieuroi
# @Site    : 
# @File    : evaluate.py
# @Software: PyCharm
import os
import sys
import torch
import pickle
import torch.nn.functional as F
from tqdm import tqdm
from MLPA import MLPA
from config import *
from evalfunc.bleu.bleu import Bleu
from evalfunc.rouge.rouge import Rouge
from evalfunc.cider.cider import Cider
from evalfunc.meteor.meteor import Meteor




def evaluate(mlpa, x, y, beam_size):
    encoder = mlpa.model.encoder
    decoder = mlpa.model.decoder

    k = beam_size


    # Encode
    encoder_out = encoder(x[0].unsqueeze(0))  # (1, encoder_dim)

    # encoder_dim = encoder_out.size(1)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, config['encoder_dim'])  # (1, 1, encoder_dim)
    # num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, 1, config['encoder_dim'])  # (k, 1, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[2]] * k).cuda()  # (k, 1) 2 represent SOS

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).cuda()  # (k, 1)

    # Lists to store completed sequences and scores
    complete_seqs = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1

    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)


        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        if config['Debug']:
            print('top_k_words:{}'.format(top_k_words))

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / config['vocab_size']  # (s)
        next_word_inds = top_k_words % config['vocab_size']  # (s)

        if config['Debug']:
            print('prev_word_inds:{}'.format(prev_word_inds))
            print('next_word_inds:{}'.format(next_word_inds))


        # Add new words to sequences
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != 3]
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
        if step > 100:
            break
        step += 1

    if len(complete_seqs_scores) > 0:
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
    else:
        seq = seqs[0]

    if config['Debug']:
        print('complete_seqs:{}'.format(complete_seqs))
        print('seq:{}'.format(seq))

    # References
    img_caps = y.tolist()[0]
    refer = [w for w in img_caps if w not in {2, 3, 0}]
    '''
    img_captions = list(
        map(lambda c: [w for w in c if w not in {2, 3, 0}],
            img_caps))  # remove <start> and pads

    references.append(img_captions)
    '''
    pred = [w for w in seq if w not in {2, 3, 0}]

    # assert len(refers) == len(preds)

    return refer, pred

    # Hypotheses
    # hypotheses.append([w for w in seq if w not in {2, 3, 0}])

    # assert len(references) == len(hypotheses)




if __name__ == "__main__":
    master_path= "./ml"
    beam_size = 2
    batch_size = 1
    inner = sys.argv[1]
    target_state = sys.argv[2]

    # training model.
    mlpa = MLPA(config)
    model_filename = "{}/{}/models_{}.pkl".format(master_path, inner, target_state)
    trained_state_dict = torch.load(model_filename)
    mlpa.load_state_dict(trained_state_dict)

    if config['use_cuda']:
        mlpa.cuda()
    mlpa.eval()

    with open('data/vocab.pickle', 'rb') as fp:
        word2idx, idx2word = pickle.load(fp)

    dataset_size = int(len(os.listdir("{}/{}".format(master_path, target_state))) / 4)

    references = dict()
    hypotheses = dict()
    refers = list()
    preds = list()
    ids = list()
    query_xs_s = []
    query_ys_s = []
    query_ids = []
    for j in list(range(dataset_size)):
        query_xs_s.append(pickle.load(open("{}/{}/query_x_{}.pkl".format(master_path, target_state, j), "rb")))
        query_ys_s.append(pickle.load(open("{}/{}/query_y_{}.pkl".format(master_path, target_state, j), "rb")))
        with open("{}/log/{}/query_x_{}_u_m_ids.txt".format(master_path, target_state, j), "r") as f:
            lines = f.readlines()
            for line in lines:
                u_id, m_id = line.strip().split('\t')
                query_ids.append('{}-{}'.format(u_id, m_id))

    num_batch = int(dataset_size // batch_size)
    if config['Debug']:
        print('num_batch:{}'.format(num_batch))

    for i in range(num_batch):
        try:
            query_xs = list(query_xs_s[batch_size * i:batch_size * (i + 1)])
            query_ys = list(query_ys_s[batch_size * i:batch_size * (i + 1)])
            query_id = list(query_ids[batch_size * i:batch_size * (i + 1)])
            batch_sz = len(query_xs)

            if config['use_cuda']:
                for i in range(batch_sz):
                    query_xs[i] = query_xs[i].cuda()
                    query_ys[i] = query_ys[i].cuda()
                    refer, pred = evaluate(mlpa, query_xs[i], query_ys[i], beam_size)
                    references[query_id[i]] = refer
                    hypotheses[query_id[i]] = pred
                    refers.append(refer)
                    preds.append(pred)
                    ids.append(query_id[i])
        except IndexError:
            continue

    assert references.keys() == hypotheses.keys()

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Cider(), "CIDEr"),
        # (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]

    newpreds = []
    for hypo in preds:
        newhypo = []
        for x in hypo:
            if not isinstance(x, int):
                if x.is_cuda:
                    x = int(x.cpu().item())
                elif x.is_tensor:
                    x = x.item()
            newhypo.append(str(x))
        newpreds.append(newhypo)

    preds = [[' '.join(hypo)] for hypo in newpreds]
    refers = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in refers]]
    # ref = [[' '.join(reft) for reft in reftmp] for reftmp in [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]


    print('refers:{}'.format(refers))
    print('preds:{}'.format(preds))
    print('idx:{}'.format(ids))

    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(refers, preds)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
    score_dict = dict(zip(method, score))

    for method, score in score_dict.items():
        print('%s:  %.4f' % (method, score))
        print("\n%s score @ beam size of %d is %.4f." % (method, beam_size, score))

