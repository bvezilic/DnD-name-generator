import os
import os.path as osp

import torch
import torch.nn as nn
from torch.optim import RMSprop
from torch.utils.data import DataLoader

from torchvision.transforms import Compose

from data import DnDCharacterNameDataset, Vocabulary, OneHot


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, input, hx, cx):
        hx, cx = self.lstm_cell(input, (hx, cx))
        logits = self.dense(self.dropout(hx))

        return logits, hx, cx


def train(epochs, hidden_size, model_name):
    vocab = Vocabulary()

    train_loder = DataLoader(dataset=DnDCharacterNameDataset(root_dir="./data",
                                                             transform=Compose([vocab,
                                                                                OneHot(len(vocab))]),
                                                             target_transform=Compose([vocab])),
                             shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rnn = RNN(input_size=len(vocab),
              hidden_size=hidden_size,
              output_size=len(vocab))
    rnn.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = RMSprop(rnn.parameters())

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch+1, epochs))
        print('-' * 10)

        optimizer.zero_grad()

        running_loss = 0
        for inputs, targets in train_loder:
            inputs = inputs.transpose(1, 0).to(device)
            targets = targets.transpose(1, 0).to(device)

            loss = 0
            hx = torch.zeros(1, hidden_size).to(device)
            cx = torch.zeros(1, hidden_size).to(device)
            for input, target in zip(inputs, targets):
                output, hx, cx = rnn(input, hx, cx)
                loss += criterion(output, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("Loss {:.4f}\n".format(running_loss/len(train_loder)))

    os.makedirs("./models", exist_ok=True)
    torch.save(rnn.to("cpu"), osp.join("models", model_name))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", default=300)
    parser.add_argument("-hs", "--hidden_size", default=64)
    parser.add_argument("-lr", "--learning_rate", default=0.0005)
    parser.add_argument("-m", "--model_name", default="model2.pt")
    args = parser.parse_args()

    train(epochs=args.epochs,
          hidden_size=args.hidden_size,
          model_name=args.model_name)
