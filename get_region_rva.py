import os
import torch
from config import get_config

from tqdm import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import RecurrentAttention

# TODO: Complete the image coordinates for dynamically obtaining scores through training the Recurrent Attention Model

def get_score_region_by_rva(frame):

    config, unparsed = get_config()

    best_model = load_checkpoint(is_best = True)
    
    x, y = x.to(config.device), y.to(config.device)
    # extract the glimpses
    for t in range(config.num_glimpses - 1):
        # forward pass through model
        h_t, l_t, b_t, p = best_model(x, l_t, h_t)

    # last iteration
    h_t, l_t, b_t, log_probas, p = best_model(x, l_t, h_t, last=True)

    # log_probas = log_probas.view(config.M, -1, log_probas.shape[-1])
    # log_probas = torch.mean(log_probas, dim=0)

    pred = get_boundingbox(log_probas)

    if pred:
        return pred[0]
    
    return None

def save_checkpoint(model_name, state, is_best):
    print('save_checkpoint')

    filename = model_name + "_ckpt.pth.tar"
    ckpt_path = os.path.join(config.ckpt_dir, filename)
    torch.save(state, ckpt_path)
    if is_best:
        filename = model_name + "_model_best.pth.tar"

def load_checkpoint(model_name, is_best = False):
    print('load_checkpoint')

    filename = model_name + "_ckpt.pth.tar"
    if is_best:
        filename = model_name + "_model_best.pth.tar"
    ckpt_path = os.path.join(config.ckpt_dir, filename)
    ckpt = torch.load(ckpt_path)

    return ckpt

def get_boundingbox():
    print('get_boundingbox')
    

class Rav_trainer:

    def __init__(self, config, data_loader):
        self.config = config

        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # glimpse network params
        self.patch_size = config.patch_size
        self.glimpse_scale = config.glimpse_scale
        self.num_patches = config.num_patches
        self.loc_hidden = config.loc_hidden
        self.glimpse_hidden = config.glimpse_hidden

        # core network params
        self.num_glimpses = config.num_glimpses
        self.hidden_size = config.hidden_size

        # reinforce params
        self.std = config.std
        self.M = config.M

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.sampler.indices)
            self.num_valid = len(self.valid_loader.sampler.indices)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
        self.num_classes = 10
        self.num_channels = 1

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr

        # misc params
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 0.0
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.plot_freq = config.plot_freq
        self.model_name = "ram_{}_{}x{}_{}".format(
            config.num_glimpses,
            config.patch_size,
            config.patch_size,
            config.glimpse_scale,
        )

        # build RAM model
        self.model = RecurrentAttention(
            self.patch_size,
            self.num_patches,
            self.glimpse_scale,
            self.num_channels,
            self.loc_hidden,
            self.glimpse_hidden,
            self.std,
            self.hidden_size,
            self.num_classes,
        )
        self.model.to(self.device)

        # initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.init_lr
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", patience=self.lr_patience
        )

    def reset(self):
        h_t = torch.zeros(
            self.batch_size,
            self.hidden_size,
            dtype=torch.float,
            device=self.device,
            requires_grad=True,
        )
        l_t = torch.FloatTensor(self.batch_size, 2).uniform_(-1, 1).to(self.device)
        l_t.requires_grad = True

        return h_t, l_t
    

    def train(self):
        print("train")

