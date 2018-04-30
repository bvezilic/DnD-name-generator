import numpy as np
from torch.utils.data import Dataset

import os.path as osp
import string


class DnDCharacterNameDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.files = ['male.txt', 'female.txt']
        self.train_data = []
        self.target_data = []

        for filename in self.files:
            with open(osp.join(root_dir, filename), 'r') as f:
                names = f.read().lower().replace(',', '').split()
                self.train_data.extend([["<EOS>"] + list(name) for name in names])
                self.target_data.extend([list(name) + ["<EOS>"] for name in names])

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, i):
        train = self.train_data[i]
        target = self.target_data[i]

        if self.transform:
            train = self.transform(train)

        if self.target_transform:
            target = self.target_transform(target)

        return train, target


class Vocabulary:
    def __init__(self):
        alphabet = string.ascii_lowercase
        self.char2idx = dict(zip(alphabet, range(1, len(alphabet) + 1)))
        self.char2idx["<EOS>"] = 0
        self.idx2char = {v: k for k, v in self.char2idx.items()}

    def __call__(self, chars):
        return np.array([self.char2idx[char] for char in chars], dtype=np.int64)

    def __len__(self):
        return len(self.char2idx)


class OneHot:
    def __init__(self, size):
        self.size = size

    def __call__(self, indexes):
        onehot = np.zeros((len(indexes), self.size))
        onehot[np.arange(len(indexes)), indexes] = 1
        return onehot.astype(np.float32)
