import os
import os.path as osp
import string

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import OneHotCategorical
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from data import DnDCharacterNameDataset, Vocabulary, OneHot, Genders, Races, ToTensor
from model import RNN


class DnDGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab = Vocabulary()
        self.races = Races()
        self.genders = Genders()
        self.onehot_vocab = OneHot(len(self.vocab))
        self.onehot_races = OneHot(len(self.races))
        self.onehot_genders = OneHot(len(self.genders))
        self.to_tensor = ToTensor()

        self.rnn = None

    def init_rnn(self, hidden_size):
        self.rnn = RNN(input_size=len(self.vocab) + len(self.races) + len(self.genders),
                       hidden_size=hidden_size,
                       output_size=len(self.vocab))
        self.rnn.to(self.device)

    def save_model(self, model_name):
        os.makedirs("./models", exist_ok=True)
        torch.save(self.rnn.to("cpu"), osp.join("models", model_name))

    def load_model(self, path):
        return torch.load(path)

    def generate(self, num_samples=5):
        with torch.no_grad():
            print("_" * 20)
            for _ in range(num_samples):
                hx, cx = self.rnn.init_states(self.device)

                letter = np.random.choice(list(string.ascii_uppercase))
                race = np.random.choice(list(self.races.races.keys()))
                gender = np.random.choice(list(self.genders.genders.keys()))

                letter_tensor = Compose([self.vocab, self.onehot_vocab, self.to_tensor])(letter).to(self.device)
                race_tensor = Compose([self.races, self.onehot_races, self.to_tensor])([race]).to(self.device)
                gender_tensor = Compose([self.genders, self.onehot_genders, self.to_tensor])([gender]).to(self.device)

                input = torch.cat([letter_tensor.to(self.device),
                                   race_tensor.to(self.device),
                                   gender_tensor.to(self.device)], 1)
                outputs = [letter]

                while True:
                    output, hx, cx = self.rnn(input, hx, cx)

                    sample = OneHotCategorical(logits=output).sample()
                    index = torch.argmax(sample)
                    char = self.vocab.idx2char[index.item()]
                    outputs.append(char)

                    input = torch.cat([sample, race_tensor, gender_tensor], 1)

                    if char == '.' or len(outputs) == 50:
                        break

                print("Start letter: {}, Race: {}, Gender: {}".format(letter, race, gender))
                print("Generated sample: {}".format(''.join(map(str, outputs))))

            print("_" * 20)

    def train(self, root_dir, epochs):
        dataset = DnDCharacterNameDataset(root_dir=root_dir,
                                          name_transform=Compose([self.vocab, self.onehot_vocab]),
                                          race_transform=Compose([self.races, self.onehot_races]),
                                          gender_transform=Compose([self.genders, self.onehot_genders]),
                                          target_transform=Compose([self.vocab]))

        train_loder = DataLoader(dataset=dataset, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = RMSprop(self.rnn.parameters())

        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch + 1, epochs))
            print('-' * 10)

            optimizer.zero_grad()

            running_loss = 0
            for inputs, targets in train_loder:
                inputs = torch.cat(list(inputs.values()), 2)
                inputs = inputs.transpose(1, 0).to(self.device)
                targets = targets.transpose(1, 0).to(self.device)

                loss = 0
                hx, cx = self.rnn.init_states(self.device)
                for input, target in zip(inputs, targets):
                    output, hx, cx = self.rnn(input, hx, cx)
                    loss += criterion(output, target)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print("Loss {:.4f}\n".format(running_loss / len(train_loder)))

            if (epoch + 1) % 2 == 0:
                self.generate()

            if (epoch + 1) % 500 == 0:
                self.save_model("epoch_{}.pt".format(epoch + 1))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", default=5000)
    parser.add_argument("-hs", "--hidden_size", default=64)
    parser.add_argument("-lr", "--learning_rate", default=0.0005)
    args = parser.parse_args()

    dnd = DnDGenerator()
    dnd.init_rnn(hidden_size=args.hidden_size)
    dnd.train(root_dir="./data",
              epochs=args.epochs)
