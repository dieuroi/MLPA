import torch
USE_CUDA = torch.cuda.is_available()

config = {
    # user/item
    'n_users': 4867,
    'n_items': 56242,
    'encoder_dim': 2048,
    'feature_dim': 16928,
    'layers': [64, 32],
    'embed_dim': 512,
    'attention_dim': 512,
    'decoder_dim': 512,
    'vocab_size': 10004,
    'dropout': 0.5,
    # cuda setting
    'use_cuda': True,
    # model setting
    'inner': 1,
    'lr': 5e-5,
    'local_lr': 5e-6,
    'batch_size': 8,
    'num_epoch': 20,
    'teacher_forcing_ratio': 1.0,
    # debug
    'Debug': False,
}


PAD_token = 0
UNK_token = 1
SOS_token = 2
EOS_token = 3
MAX_LENGTH = 100
save_dir = '/home/PAC/aNet/data'
states = ["warm_state", "user_cold_state", "item_cold_state", "user_and_item_cold_state"]
