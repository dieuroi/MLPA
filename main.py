import os
import random
import torch
import pickle

from MLPA import MLPA
from config import *
from data_generation import generate

def training(mlpa, total_dataset, batch_size, num_epoch, model_save=True, model_filename=None):
    if config['use_cuda']:
        mlpa.cuda()

    training_set_size = len(total_dataset)
    mlpa.train()
    for _ in range(num_epoch):
        random.shuffle(total_dataset)
        num_batch = int(training_set_size // batch_size)
        a,b,c,d = zip(*total_dataset)
        for i in range(num_batch):
            try:
                supp_xs = list(a[batch_size*i:batch_size*(i+1)])
                supp_ys = list(b[batch_size*i:batch_size*(i+1)])
                query_xs = list(c[batch_size*i:batch_size*(i+1)])
                query_ys = list(d[batch_size*i:batch_size*(i+1)])
            except IndexError:
                continue
            mlpa.global_update(supp_xs, supp_ys, query_xs, query_ys, config['inner'])

    if model_save:
        torch.save(mlpa.state_dict(), model_filename)


if __name__ == "__main__":
    master_path= "./ml"
    if not os.path.exists("{}/".format(master_path)):
        os.mkdir("{}/".format(master_path))
        # preparing dataset. It needs about 22GB of your hard disk space.
    generate(master_path)
    print('Data generation is done!')

    # training model.
    mlpa = MLPA(config)
    model_filename = "{}/models.pkl".format(master_path)
    if not os.path.exists(model_filename):
        # Load training dataset.
        training_set_size = int(len(os.listdir("{}/warm_state".format(master_path))) / 4)
        supp_xs_s = []
        supp_ys_s = []
        query_xs_s = []
        query_ys_s = []
        for idx in range(training_set_size):
            supp_xs_s.append(pickle.load(open("{}/warm_state/supp_x_{}.pkl".format(master_path, idx), "rb")))
            supp_ys_s.append(pickle.load(open("{}/warm_state/supp_y_{}.pkl".format(master_path, idx), "rb")))
            query_xs_s.append(pickle.load(open("{}/warm_state/query_x_{}.pkl".format(master_path, idx), "rb")))
            query_ys_s.append(pickle.load(open("{}/warm_state/query_y_{}.pkl".format(master_path, idx), "rb")))
        total_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))
        del(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)
        training(mlpa, total_dataset, batch_size=config['batch_size'], num_epoch=config['num_epoch'], model_save=True, model_filename=model_filename)
    else:
        trained_state_dict = torch.load(model_filename)
        mlpa.load_state_dict(trained_state_dict)