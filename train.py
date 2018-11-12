import torch.nn as nn
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from data import DnDCharacterNameDataset, Vocabulary, OneHot, Genders, Races, ToTensor
from model import RNNCellModel, RNNLayerModel
from utils import save_model


class Trainer:
    def __init__(self, root_dir, hidden_size, lr, epochs, batch_size, device):
        self.root_dir = root_dir
        self.device = device

        # Training params
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        # Model params
        self.hidden_size = hidden_size

        # Data params
        self.vocab = Vocabulary()
        self.races = Races()
        self.genders = Genders()

        # Initialization
        self.dataset = self.init_dataset()
        self.train_loder = self.init_loader()
        self.model = self.init_model()
        self.criterion = self.init_criterion()
        self.optimizer = self.init_optimizer()

    def init_dataset(self):
        raise NotImplementedError

    def init_loader(self):
        raise NotImplementedError

    def init_model(self):
        raise NotImplementedError

    def init_criterion(self):
        raise NotImplementedError

    def init_optimizer(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


class RNNCellTrainer(Trainer):
    def __init__(self, root_dir, hidden_size=128, lr=0.0005, epochs=50, batch_size=512, device='gpu'):
        super().__init__(root_dir, hidden_size, lr, epochs, batch_size, device)

    def init_dataset(self):
        return DnDCharacterNameDataset(root_dir=self.root_dir,
                                       name_transform=Compose([self.vocab, OneHot(self.vocab.size), ToTensor()]),
                                       race_transform=Compose([self.races, OneHot(self.races.size), ToTensor()]),
                                       gender_transform=Compose([self.genders, OneHot(self.genders.size), ToTensor()]),
                                       target_transform=Compose([self.vocab, ToTensor()]))

    def init_loader(self):
        return DataLoader(dataset=self.dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=lambda batch: self.dataset.collate_fn(batch))

    def init_model(self):
        model = RNNCellModel(input_size=self.vocab.size + self.races.size + self.genders.size,
                             hidden_size=self.hidden_size,
                             output_size=self.vocab.size)
        model.to(self.device)
        return model

    def init_criterion(self):
        return nn.CrossEntropyLoss(ignore_index=-1)

    def init_optimizer(self):
        return RMSprop(self.model.parameters())

    def train(self):
        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(epoch + 1, self.epochs))
            print('-' * 10)

            self.optimizer.zero_grad()

            running_loss = 0
            for inputs, targets, _ in self.train_loder:
                batch_size = inputs.shape[1]

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                loss = 0
                hx, cx = self.model.init_states(batch_size=batch_size, device=self.device)
                for input, target in zip(inputs, targets):
                    output, hx, cx = self.model(input, hx, cx)
                    loss += self.criterion(output, target)

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            print("Loss {:.4f}\n".format(running_loss / len(self.train_loder)))

        save_model(self.model, "rnn_cell_epoch_{}.pt".format(self.epochs))


class RNNLayerTrainer(Trainer):
    def __init__(self, root_dir, hidden_size=128, lr=0.0005, epochs=50, batch_size=512, device='gpu'):
        super().__init__(root_dir, hidden_size, lr, epochs, batch_size, device)

    def init_dataset(self):
        return DnDCharacterNameDataset(root_dir=self.root_dir,
                                       name_transform=Compose([self.vocab, OneHot(self.vocab.size), ToTensor()]),
                                       race_transform=Compose([self.races, OneHot(self.races.size), ToTensor()]),
                                       gender_transform=Compose([self.genders, OneHot(self.genders.size), ToTensor()]),
                                       target_transform=Compose([self.vocab, ToTensor()]))

    def init_loader(self):
        return DataLoader(dataset=self.dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=lambda batch: self.dataset.collate_fn(batch))

    def init_model(self):
        model = RNNLayerModel(input_size=self.vocab.size + self.races.size + self.genders.size,
                              hidden_size=self.hidden_size,
                              output_size=self.vocab.size)
        model.to(self.device)
        return model

    def init_criterion(self):
        return nn.CrossEntropyLoss(ignore_index=-1)

    def init_optimizer(self):
        return RMSprop(self.model.parameters())

    def train(self):
        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(epoch + 1, self.epochs))
            print('-' * 10)

            self.optimizer.zero_grad()

            running_loss = 0
            for inputs, targets, lengths in self.train_loder:
                batch_size = inputs.shape[1]

                inputs = inputs.to(self.device)  # shape: [T, B, *]
                targets = targets.to(self.device)  # shape: [T, B]

                h0, c0 = self.model.init_states(batch_size=batch_size, device=self.device)
                output, hx, cx = self.model(inputs, h0, c0, lengths)

                loss = self.criterion(output.view(-1, output.shape[-1]), targets.view(-1))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print("Loss {:.4f}\n".format(running_loss / len(self.train_loder)))

        save_model(self.model, "rnn_layer_epoch_{}.pt".format(self.epochs))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", default=100)
    parser.add_argument("-bs", "--batch_size", default=512)
    parser.add_argument("-hs", "--hidden_size", default=64)
    parser.add_argument("-lr", "--learning_rate", default=0.0001)
    parser.add_argument("-d", "--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("-t", "--type", default="cell", choices=["cell", "layer"])
    args = parser.parse_args()

    if args.type == "layer":
        trainer = RNNLayerTrainer(root_dir="./data",
                                  epochs=args.epochs,
                                  batch_size=args.batch_size,
                                  lr=args.learning_rate,
                                  device=args.device)
    else:
        trainer = RNNCellTrainer(root_dir="./data",
                                 epochs=args.epochs,
                                 batch_size=args.batch_size,
                                 lr=args.learning_rate,
                                 device=args.device)
    trainer.train()
