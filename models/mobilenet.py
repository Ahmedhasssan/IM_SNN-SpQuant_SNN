"""
MobileNet Spiking Neural Networks
"""
import torch
import torch.nn as nn
from .spikes import SeqToANNContainer
from .layers import SConv, SConvDW
from .t2c.methods import QLinear

class MBNETSNN(nn.Module):
    def __init__(self, num_classes=10, wbit=4, tau=0.5):
        super(MBNETSNN, self).__init__()
        self.features = nn.Sequential(
            SConv(3, 32, 3, 1, 1, pool=True, wbit=wbit, tau=tau),
            SConvDW(32, 64, 3, 1, 1, pool=True, wbit=wbit, tau=tau),
            SConvDW(64, 64, 3, 1, 1, pool=True, wbit=wbit, tau=tau),
            SConvDW(64, 128, 3, 1, 1, pool=True, wbit=wbit, tau=tau),
            SConvDW(128, 128, 3, 1, 1, pool=False, wbit=wbit, tau=tau),
        )

        W = int(48/2/2/2/2)
        if wbit < 32:
            self.classifier1 = SeqToANNContainer(QLinear(1152, num_classes, wbit=wbit))
        else:
            self.classifier1 = SeqToANNContainer(nn.Linear(1152, num_classes))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        input = input/input.max().item()
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier1(x)
        return x
    
def mobilenet_tiny(num_classes=10, wbit=4, tau=0.5):
    model = MBNETSNN(num_classes=num_classes, wbit=wbit, tau=tau)
    return model