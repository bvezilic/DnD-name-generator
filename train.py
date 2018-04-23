import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import Compose
from data import DnDCharacterNameDataset, Vocabulary, OneHot


def train(args):
    vocab = Vocabulary()

    train_loder = DataLoader(dataset=DnDCharacterNameDataset(root_dir="./data",
                                                             transform=Compose([vocab,
                                                                                OneHot(len(vocab))]),
                                                             target_transform=Compose([vocab])))

    rnn = nn.LSTMCell(input_size=len(vocab),
                      hidden_size=args.hidden_size)

    optimizer = RMSprop(rnn.parameters())

    for epoch in range(args.epochs):
        print("Epoch {}/{}".format(epoch+1, args.epochs))
        print('-' * 10)
        optimizer.zero_grad()

        running_loss = 0
        for inputs, targets in train_loder:
            loss = 0
            hx = Variable(torch.randn(1, args.hidden_size))
            cx = Variable(torch.randn(1, args.hidden_size))
            for input, target in zip(inputs[0], targets[0]):
                input, taget = Variable(input).float(), Variable(torch.LongTensor([target]))

                hx, cx = rnn(input, (hx, cx))
                loss += F.cross_entropy(hx, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

        print("Loss {:.4f}\n".format(running_loss/len(train_loder)))

    os.makedirs("./models", exist_ok=True)
    torch.save(rnn, "models/" + args.model_name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default=100)
    parser.add_argument('-hs', '--hidden_size', default=128)
    parser.add_argument('-m', '--model_name', default='model_cuda.pt')
    args = parser.parse_args()

    train(args)
