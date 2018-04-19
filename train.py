import torch
import torch.nn as nn

from torch.utils.data import DataLoader


rnn = nn.LSTMCell(input_size=1,
                  hidden_size=32)
