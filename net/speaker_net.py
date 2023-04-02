from net.layers import *


class SpeakerNet(nn.Module):
    def __init__(self, config, number_of_labels):
        super().__init__()
        # self.mfcc = MFCC()
        config[-1]['params']['out_features'] = number_of_labels
        self.layers = nn.ModuleList()

        for layer_info in config:
            layer_class = name_to_layer[layer_info['name']]
            current_layer = layer_class(**layer_info['params'])
            self.layers.append(current_layer)

    def forward(self, x):
        # x = self.mfcc(x)
        for layer in self.layers:
            x = layer(x)
        return x
