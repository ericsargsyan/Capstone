import torch
from torch import nn
from torchaudio.transforms import MFCC


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
        self.do_relu = relu

    def forward(self, x):
        if self.do_relu:
            return nn.functional.relu(self.func(x))
        return self.func(x)


class BBlock(nn.Module):
    def __init__(self, input_size, kernel_size, output_size, repeat):
        super().__init__()
        self.model = nn.ModuleList([TSConv(input_size, kernel_size, output_size)])
        for i in range(repeat-2):
            self.model.append(TSConv(output_size, kernel_size, output_size))
        self.model.append(TSConv(output_size, kernel_size, output_size, relu=False))

        self.repeat = repeat
        self.pointwise = nn.Conv1d(input_size, output_size, kernel_size=1)
        self.batchnorm = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = x
        for i in range(self.repeat):
            x1 = self.model[i](x1)

        x2 = self.batchnorm(self.pointwise(x))
        return self.relu(x1 + x2)

class ConvBatchNormRelu(nn.Module):
    def __init__(self, input_size, kernel_size, output_size):
        super().__init__()
        self.func = nn.Sequential(
            nn.Conv1d(input_size, output_size, kernel_size=kernel_size),
            nn.BatchNorm1d(output_size)
        )

    def forward(self, x):
        return self.func(x)


class StatsPooling(nn.Module):
    def forward(self, x):
        x = torch.cat([torch.mean(x, dim=-1), torch.std(x, dim=-1)], dim=1)
        return x


name_to_layer = nn.__dict__
name_to_layer["StatsPooling"] = StatsPooling
name_to_layer["ConvBatchNormRelu"] = ConvBatchNormRelu
name_to_layer["BBlock"] = BBlock
name_to_layer["MFCC"] = MFCC
