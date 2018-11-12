import string

import numpy as np
import torch
from torch.distributions import OneHotCategorical
from torchvision.transforms import Compose

from data import Vocabulary, OneHot, Genders, Races, ToTensor
from utils import load_model


class Generator:
    def __init__(self, model_path, device="cpu"):
        self.model = load_model(model_path, device=device)
        self.device = device

    def generate(self, num_samples):
        raise NotImplementedError


class RNNCellGenerator(Generator):
    def __init__(self, model_path, device="cpu"):
        super().__init__(model_path, device)

        self.vocab = Vocabulary()
        self.races = Races()
        self.genders = Genders()
        self.to_tensor = ToTensor()

        self.name_transform = Compose([self.vocab, OneHot(self.vocab.size), ToTensor()])
        self.race_transform = Compose([self.races, OneHot(self.races.size), ToTensor()])
        self.gender_transform = Compose([self.genders, OneHot(self.genders.size), ToTensor()])

    def generate(self, num_samples):
        with torch.no_grad():
            print("_" * 20)
            for _ in range(num_samples):
                hx, cx = self.model.init_states(batch_size=1, device=self.device)

                letter = np.random.choice(self.vocab.start_letters)
                race = np.random.choice(self.races.available_races)
                gender = np.random.choice(self.genders.available_genders)

                letter_tensor = self.name_transform(letter).to(self.device)
                race_tensor = self.race_transform(race).to(self.device)
                gender_tensor = self.gender_transform(gender).to(self.device)

                input = torch.cat([letter_tensor, race_tensor, gender_tensor], 1)
                outputs = [letter]

                while True:
                    output, hx, cx = self.model(input, hx, cx)

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


class RNNLayerGenerator(Generator):
    def __init__(self, model_path, device="cpu"):
        super().__init__(model_path, device)

        self.vocab = Vocabulary()
        self.races = Races()
        self.genders = Genders()
        self.to_tensor = ToTensor()

        self.name_transform = Compose([self.vocab, OneHot(self.vocab.size), ToTensor()])
        self.race_transform = Compose([self.races, OneHot(self.races.size), ToTensor()])
        self.gender_transform = Compose([self.genders, OneHot(self.genders.size), ToTensor()])

    def generate(self, num_samples):
        with torch.no_grad():
            print("_" * 20)
            for _ in range(num_samples):
                hx, cx = self.model.init_states(batch_size=1, device=self.device)

                letter = np.random.choice(self.vocab.start_letters)
                race = np.random.choice(self.races.available_races)
                gender = np.random.choice(self.genders.available_genders)

                letter_tensor = self.name_transform(letter).to(self.device)
                race_tensor = self.race_transform(race).to(self.device)
                gender_tensor = self.gender_transform(gender).to(self.device)

                letter_tensor = torch.unsqueeze(letter_tensor, 0)
                race_tensor = torch.unsqueeze(race_tensor, 0)
                gender_tensor = torch.unsqueeze(gender_tensor, 0)

                input = torch.cat([letter_tensor, race_tensor, gender_tensor], 2)
                outputs = [letter]

                while True:
                    output, hx, cx = self.model(input, hx, cx, torch.tensor([1]))

                    sample = OneHotCategorical(logits=output).sample()
                    index = torch.argmax(sample)
                    char = self.vocab.idx2char[index.item()]
                    outputs.append(char)

                    input = torch.cat([sample, race_tensor, gender_tensor], 2)

                    if char == '.' or len(outputs) == 50:
                        break

                print("Start letter: {}, Race: {}, Gender: {}".format(letter, race, gender))
                print("Generated sample: {}".format(''.join(map(str, outputs))))

            print("_" * 20)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-mp", "--model_path")
    args = parser.parse_args()

    dnd = RNNLayerGenerator(model_path="./models/rnn_layer_epoch_100.pt")
    dnd.generate(5)
