import torch
import torch.nn as nn
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from data import DnDCharacterNameDataset, Vocabulary, OneHot, Genders, Races, ToTensor
from model import RNN
from utils import save_model


class RNNTrainer:
    def __init__(self, root_dir, lr=0.0005, epochs=500, batch_size=32):
        self.root_dir = root_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.vocab = Vocabulary()
        self.races = Races()
        self.genders = Genders()

        self.rnn = None

    def init_rnn(self, hidden_size):
        self.rnn = RNN(input_size=len(self.vocab) + len(self.races) + len(self.genders),
                       hidden_size=hidden_size,
                       output_size=len(self.vocab))
        self.rnn.to(self.device)

    def train(self):
        dataset = DnDCharacterNameDataset(root_dir=self.root_dir,
                                          name_transform=Compose([self.vocab, OneHot(self.vocab.size), ToTensor()]),
                                          race_transform=Compose([self.races, OneHot(self.races.size), ToTensor()]),
                                          gender_transform=Compose([self.genders, OneHot(self.genders.size), ToTensor()]),
                                          target_transform=Compose([self.vocab, ToTensor()]))

        train_loder = DataLoader(dataset=dataset,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 collate_fn=dataset.collate_fn)

        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = RMSprop(self.rnn.parameters())

        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(epoch + 1, self.epochs))
            print('-' * 10)

            optimizer.zero_grad()

            running_loss = 0
            for inputs, targets in train_loder:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                loss = 0
                hx, cx = self.rnn.init_states(batch_size=inputs.shape[1], device=self.device)
                for input, target in zip(inputs, targets):
                    output, hx, cx = self.rnn(input, hx, cx)
                    loss += criterion(output, target)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print("Loss {:.4f}\n".format(running_loss / len(train_loder)))

            if (epoch + 1) % 20 == 0:
                save_model("epoch_{}.pt".format(epoch + 1))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", default=500)
    parser.add_argument("-bs", "--batch_size", default=512)
    parser.add_argument("-hs", "--hidden_size", default=64)
    parser.add_argument("-lr", "--learning_rate", default=0.0005)
    args = parser.parse_args()

    trainer = RNNTrainer(root_dir="./data",
                         epochs=args.epochs,
                         batch_size=args.batch_size,
                         lr=args.learning_rate)
    trainer.init_rnn(hidden_size=args.hidden_size)
    trainer.train()
