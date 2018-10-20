import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
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
