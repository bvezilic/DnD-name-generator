import os.path as osp
from torch.distributions import Categorical
import torch


def sample(model_name):
    rnn = torch.load(osp.join('models', model_name))

    with torch.no_grad():
        pass


if __name__ == '__main__':
    sample()
