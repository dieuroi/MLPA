
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/22 10:11
# @Author  : dieuroi
# @Site    : 
# @File    : evaluate_nn.py
# @Software: PyCharm
import os
from config import *
from ast import literal_eval
from evalfunc.bleu.bleu import Bleu
from evalfunc.rouge.rouge import Rouge
from evalfunc.cider.cider import Cider
from evalfunc.meteor.meteor import Meteor

scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Cider(), "CIDEr"),
        # (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]

refers = []
preds = []

with open('cold-start.txt','r') as f:
    lines = f.readlines()
    lines = [literal_eval(line) for line in lines]

    for i, line in enumerate(lines):
        if i%2 == 0:
            refers.append(line)
        else:
            preds.append(line)

for i in range(len(refers)):
    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(refers[i], preds[i])
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
    score_dict = dict(zip(method, score))
    for method, score in score_dict.items():
        print('%s:  %.4f' % (method, score))
        print("\n%s score @ beam size of %d is %.4f." % (method, 1, score))