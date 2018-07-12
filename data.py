import glob
import os.path as osp
import string

import numpy as np
from torch.utils.data import Dataset


class DnDCharacterNameDataset(Dataset):
    def __init__(self, root_dir, name_transform=None, race_transform=None, gender_transform=None,
                 target_transform=None, end_char='.'):
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
        train = self.train_data[i]
        target = self.target_data[i]

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
        samples = []
        for i in range(5):
            samples.append(self.train_data[i])

        return str(samples)


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


class Genders:
    def __init__(self):
        self.genders = {'male': 0, 'female': 1}

    def __getitem__(self, item):
        return self.genders.get(item)

    def __len__(self):
        return len(self.genders)

    def __call__(self, items):
        return [self.genders.get(item) for item in items]


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


class OneHot:
    def __init__(self, size):
        self.size = size

    def __call__(self, indexes):
        onehot = np.zeros((len(indexes), self.size), dtype=np.float32)
        onehot[np.arange(len(indexes)), indexes] = 1
        return onehot


if __name__ == '__main__':
    dataset = DnDCharacterNameDataset(root_dir='data')
    print(dataset)
