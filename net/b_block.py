from torch import nn
from ts_conv import *


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

x = torch.rand((165*40, 165*40))
model = BBlock(165*40, 3, 165*40, 3)
print(model.forward(x))