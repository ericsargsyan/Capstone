import pytorch_lightning
import torch
from torch import nn
import torchaudio


class TSConv(nn.Module):
    def __init__(self, input_size, kernel_size, output_size, relu=True):
        super().__init__()
        self.func = nn.Sequential(
            nn.Conv1d(in_channels=input_size,
                      out_channels=input_size,
                      kernel_size=kernel_size,
                      groups=input_size,
                      padding="same"),
            nn.Conv1d(input_size, output_size, kernel_size=1),
            nn.BatchNorm1d(output_size)
        )
        self.du_relu = relu

    def forward(self, x):
        if self.du_relu:
            return nn.functional.relu(self.func(x))
        return self.func(x)

# x = torch.randn(165*40, 165*40)
# conv = nn.Conv1d(10, 8, 2)
# print(conv(x))
# model = TSconv(165*40, 3, 165*40)
# print(model.forward(x).size())
