from torch.utils.data import Dataset
import os.path as osp


class DnDCharacterNameDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = ['male.txt', 'female.txt']
        self.train_data = []

        for filename in self.files:
            with open(osp.join(root_dir, filename), 'r') as f:
                names = f.read().lower().replace(',', '').split()
                self.train_data.extend(names)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        if self.transform:
            item = self.transform(item)

        return item


class Vocab:
    def __init__(self):
        self.char2idx = {}

        idx = 1
        for name in self.train_data:
            for c in name:
                if c not in self.char2idx:
                    self.char2idx['c'] = idx
                    idx += 1

        self.idx2char = {v: k for k, v in self.char2idx.items()}

    def to_idx(self, name):
        raise NotImplementedError

    def to_char(self, name):
        raise NotImplementedError


class OneHot:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError
