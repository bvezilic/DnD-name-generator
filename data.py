import copy
import glob
import os.path as osp
import string

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_sequence
from torch.utils.data import Dataset


class DnDCharacterNameDataset(Dataset):
    def __init__(self, root_dir,
                 name_transform=None,
                 race_transform=None,
                 gender_transform=None,
                 target_transform=None,
                 end_char='.'):

        self.root_dir = root_dir
        self.name_transform = name_transform
        self.race_transform = race_transform
        self.gender_transform = gender_transform
        self.target_transform = target_transform
        self.train_data = []
        self.target_data = []

        for filename in glob.glob(osp.join(root_dir, '*.txt')):
            race, gender = osp.basename(osp.splitext(filename)[0]).split('_')
            with open(filename, 'r') as f:
                names = f.read().replace(',', '').split()
                for name in names:
                    self.train_data.append({'name': list(name),
                                            'race': [race] * len(name),
                                            'gender': [gender] * len(name)})
                    self.target_data.append(list(name[1:]) + [end_char])

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, i):
        train = copy.deepcopy(self.train_data[i])
        target = copy.deepcopy(self.target_data[i])

        if self.name_transform:
            train['name'] = self.name_transform(train['name'])

        if self.race_transform:
            train['race'] = self.race_transform(train['race'])

        if self.gender_transform:
            train['gender'] = self.gender_transform(train['gender'])

        if self.target_transform:
            target = self.target_transform(target)

        return train, target

    def __str__(self):
        samples = [self.train_data[i] for i in range(5)]
        return str(samples)

    @staticmethod
    def collate_fn(batch):
        batch = sorted(batch, key=lambda x: x[1].shape[0], reverse=True)

        inputs, targets = zip(*batch)
        inputs = [torch.cat([sample['name'], sample['race'], sample['gender']], 1) for sample in inputs]
        lengths = [input.shape[0] for input in inputs]

        inputs = pad_sequence(inputs, padding_value=0, batch_first=True)
        targets = pad_sequence(targets, padding_value=-1, batch_first=True)

        return inputs, targets, lengths


class Races:
    def __init__(self):
        self.available_races = ['human', 'elf', 'dwarf', 'halforc', 'halfling', 'tiefling', 'dragonborn']
        self.races = dict(zip(self.available_races, np.arange(len(self.available_races))))

    def __len__(self):
        return len(self.races)

    def __getitem__(self, item):
        return self.races.get(item)

    def __call__(self, items):
        return [self.races.get(item) for item in items]

    @property
    def size(self):
        return len(self)


class Genders:
    def __init__(self):
        self.genders = {'male': 0, 'female': 1}

    def __getitem__(self, item):
        return self.genders.get(item)

    def __len__(self):
        return len(self.genders)

    def __call__(self, items):
        return [self.genders.get(item) for item in items]

    @property
    def size(self):
        return len(self)


class Vocabulary:
    def __init__(self, end_char='.'):
        alphabet = string.ascii_letters + '-'
        self.char2idx = dict(zip(alphabet, range(1, len(alphabet) + 1)))
        self.char2idx[end_char] = 0
        self.idx2char = {v: k for k, v in self.char2idx.items()}

    def __len__(self):
        return len(self.char2idx)

    def __getitem__(self, item):
        return self.char2idx.get(item)

    def __call__(self, chars):
        return np.array([self.char2idx[char] for char in chars], dtype=np.int64)

    @property
    def size(self):
        return len(self)


class OneHot:
    def __init__(self, size):
        self.size = size

    def __call__(self, indexes):
        onehot = np.zeros((len(indexes), self.size), dtype=np.float32)
        onehot[np.arange(len(indexes)), indexes] = 1
        return onehot


class ToTensor:
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, data):
        return torch.tensor(data, dtype=self.dtype)
