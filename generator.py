import string

import numpy as np
import torch
from torch.distributions import OneHotCategorical
from torchvision.transforms import Compose

from data import Vocabulary, OneHot, Genders, Races, ToTensor
from utils import load_model


class DnDGenerator:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.rnn = load_model(model_path, device=device)

        self.vocab = Vocabulary()
        self.races = Races()
        self.genders = Genders()
        self.to_tensor = ToTensor()

        self.name_transform = Compose([self.vocab, OneHot(self.vocab.size), ToTensor()])
        self.race_transform = Compose([self.races, OneHot(self.races.size), ToTensor()])
        self.gender_transform = Compose([self.genders, OneHot(self.genders.size), ToTensor()])

    def generate(self, num_samples=5):
        with torch.no_grad():
            print("_" * 20)
            for _ in range(num_samples):
                hx, cx = self.rnn.init_states(batch_size=1, device=self.device)

                letter = np.random.choice(list(string.ascii_uppercase))
                race = np.random.choice(list(self.races.races.keys()))
                gender = np.random.choice(list(self.genders.genders.keys()))

                letter_tensor = self.name_transform(letter).to(self.device)
                race_tensor = self.race_transform([race]).to(self.device)
                gender_tensor = self.gender_transform([gender]).to(self.device)

                input = torch.cat([letter_tensor, race_tensor, gender_tensor], 1)
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-mp", "--model_path")
    args = parser.parse_args()

    dnd = DnDGenerator(model_path="./models/epoch_500.pt")
    dnd.generate(5)
