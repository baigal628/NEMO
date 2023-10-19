# Create conv model

import torch
import torch.nn as nn
from torch.autograd import Variable

class NanoporeConvNet(nn.Module):

    def __init__(self, input_size=400, hidden_size=256):
        super(NanoporeConvNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=3, stride=1),
            nn.ReLU(),
            #nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2),
            nn.ReLU(),
            #nn.MaxPool1d(kernel_size=2),#
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2),
            nn.ReLU(),
            #nn.MaxPool1d(kernel_size=2),#
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2),#
            nn.ReLU(),#
            #nn.MaxPool1d(kernel_size=2),#
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1))
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):

        conv_out = self.model((x - 90.) / 40.)
        pool_out, _ = conv_out.max(axis=-1)
        label_out = self.linear(pool_out)
        #squashed_out = torch.sigmoid(label_out)

        return label_out
