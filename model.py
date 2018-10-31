import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class RNNCellModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, input, hx, cx):
        hx, cx = self.lstm_cell(input, (hx, cx))
        logits = self.dense(self.dropout(hx))

        return logits, hx, cx

    def init_states(self, batch_size, device):
        hx = torch.zeros(batch_size, self.hidden_size).to(device)
        cx = torch.zeros(batch_size, self.hidden_size).to(device)

        return hx, cx


class RNNLayerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hx, cx, lengths):
        inputs = pack_padded_sequence(inputs, lengths=lengths)
        outputs, (h_n, c_n) = self.lstm(inputs, (hx, cx))
        pad_outputs, _ = pad_packed_sequence(outputs, padding_value=-1)
        logits = self.dense(self.dropout(pad_outputs))

        return logits, h_n, c_n

    def init_states(self, batch_size, device):
        hx = torch.zeros(1, batch_size, self.hidden_size).to(device)
        cx = torch.zeros(1, batch_size, self.hidden_size).to(device)

        return hx, cx
