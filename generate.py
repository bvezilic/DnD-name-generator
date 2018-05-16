import os.path as osp
from torch.distributions import OneHotCategorical
import torch

from data import Vocabulary, OneHot
from train import RNN


def generate(model_name):
    vocab = Vocabulary()
    one_hot = OneHot(len(vocab))

    rnn = torch.load(osp.join('models', model_name))
    rnn.to('cpu')
    rnn.eval()

    with torch.no_grad():
        hx = torch.zeros(1, rnn.lstm_cell.hidden_size)
        cx = torch.zeros(1, rnn.lstm_cell.hidden_size)
        input = torch.tensor(one_hot([21]))
        outputs = [vocab.idx2char[torch.argmax(input).item()]]

        while True:
            output, hx, cx = rnn(input, hx, cx)

            sample = OneHotCategorical(logits=output).sample()
            input = sample
            index = torch.argmax(sample)
            char = vocab.idx2char[index.item()]
            outputs.append(char)

            if char == '\n' or len(outputs) == 50:
                break

    print("Generated name:")
    print(''.join(map(str, outputs)))


if __name__ == '__main__':
    generate(model_name='model.pt')
