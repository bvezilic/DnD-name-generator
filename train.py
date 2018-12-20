import logging

import torch.nn as nn
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from data import DnDCharacterNameDataset, Vocabulary, OneHot, Genders, Races, ToTensor
from model import RNNCellModel, RNNLayerModel
from utils import save_model


class Trainer:
    """
    Base Trainer class that describes basic set attributes and methods needed to run model training.

    Every subclass needs to implement following methods:
        - init_dataset
        - init_loader
        - init_model
        - init_criterion
        - init_optimizer
    """
    def __init__(self, root_dir, hidden_size, lr, epochs, batch_size, device, logfile):
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

        # Initialize logging
        logging.basicConfig(filename=logfile, filemode='w', level=0, format="%(message)s")

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

    def run_train_loop(self):
        raise NotImplementedError


class RNNCellTrainer(Trainer):
    """
    Trainer class for training the LSTMCell model (RNNCellModel). Defines methods for:
        - Initializing dataset
        - Initializing data loader
        - Initializing model
        - Initializing criterion
        - Initializing optimizer
    """
    def __init__(self, root_dir,
                 hidden_size=128,
                 lr=0.0005,
                 epochs=50,
                 batch_size=512,
                 device='gpu',
                 logfile='train_loss.log'):
        super().__init__(root_dir, hidden_size, lr, epochs, batch_size, device, logfile)

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
        """
        Due to variable sequence variable length, output tensor will contain -1 value. Time-steps that contain -1
        value as target (y) will not be included in loss function
        """
        return nn.CrossEntropyLoss(ignore_index=-1)

    def init_optimizer(self):
        return RMSprop(self.model.parameters())

    def run_train_loop(self):
        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(epoch + 1, self.epochs))
            print('-' * 10)

            self.optimizer.zero_grad()

            running_loss = 0
            for inputs, targets, _ in self.train_loder:
                batch_size = inputs.shape[1]

                inputs = inputs.to(self.device)  # shape: [T, B, *]
                targets = targets.to(self.device)  # shape: [T, B]

                loss = 0
                hx, cx = self.model.init_states(batch_size=batch_size, device=self.device)

                # Iterate over time-steps and add loss
                for input, target in zip(inputs, targets):
                    output, hx, cx = self.model(input, hx, cx)
                    loss += self.criterion(output, target)

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(self.train_loder)
            logging.info("Epoch: {}, Loss: {}".format(epoch + 1, epoch_loss))
            print("Loss {:.4f}\n".format(epoch_loss))

            if epoch+1 in (1, 5, 10, 25):
                save_model(self.model, "rnn_cell_epoch_{}.pt".format(epoch+1))


class RNNLayerTrainer(Trainer):
    """
    Trainer class for training the LSTMLayer model (RNNLayerModel). Defines methods for:
        - Initializing dataset
        - Initializing data loader
        - Initializing model
        - Initializing criterion
        - Initializing optimizer
    """
    def __init__(self, root_dir,
                 hidden_size=128,
                 lr=0.0005,
                 epochs=50,
                 batch_size=512,
                 device='gpu',
                 logfile='train_loss.log'):
        super().__init__(root_dir, hidden_size, lr, epochs, batch_size, device, logfile)

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
        """
        Due to variable sequence variable length, output tensor will contain -1 value. Time-steps that contain -1
        value as target (y) will not be included in loss function
        """
        return nn.CrossEntropyLoss(ignore_index=-1)

    def init_optimizer(self):
        return RMSprop(self.model.parameters())

    def run_train_loop(self):
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

            epoch_loss = running_loss / len(self.train_loder)
            logging.info("Epoch: {}, Loss: {}".format(epoch+1, epoch_loss))
            print("Loss {:.4f}\n".format(epoch_loss))

            if epoch+1 in (1, 5, 10, 25, 50, 75, 100):
                save_model(self.model, "rnn_layer_epoch_{}.pt".format(epoch+1))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", default=100)
    parser.add_argument("-bs", "--batch_size", default=128)
    parser.add_argument("-hs", "--hidden_size", default=64)
    parser.add_argument("-lr", "--learning_rate", default=0.0001)
    parser.add_argument("-d", "--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("-t", "--type", default="layer", choices=["cell", "layer"])
    parser.add_argument("-l", "--logfile", default="train_loss.log")
    args = parser.parse_args()

    if args.type == "layer":
        trainer = RNNLayerTrainer(root_dir="./data",
                                  epochs=args.epochs,
                                  batch_size=args.batch_size,
                                  lr=args.learning_rate,
                                  device=args.device,
                                  logfile=args.logfile)
    else:
        trainer = RNNCellTrainer(root_dir="./data",
                                 epochs=args.epochs,
                                 batch_size=args.batch_size,
                                 lr=args.learning_rate,
                                 device=args.device,
                                 logfile=args.logfile)
    trainer.run_train_loop()
