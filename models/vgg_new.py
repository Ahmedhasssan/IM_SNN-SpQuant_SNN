"""
Spiking Neural Network with VGG architecture 
"""

import torch.nn as nn
from .spikes import *
from .layers import SConv
from .cg import *
from .layer import *

class VGGSNN(nn.Module):
    def __init__(self, num_classes, baseline = True):
        super(VGGSNN, self).__init__()
        self.num_classes = num_classes
        self.baseline = baseline
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.features = nn.Sequential(
            SConv(3,64,3,1,1),
            SConv(64,128,3,1,1),
            pool,
            SConv(128,256,3,1,1),
            SConv(256,256,3,1,1),
            pool,
            SConv(256,512,3,1,1),
            SConv(512,512,3,1,1),
            pool,
            SConv(512,512,3,1,1),
            SConv(512,512,3,1,1),
            pool,
        )
        W = int(48/2/2/2/2)
        if num_classes ==10:
            self.classifier = SeqToANNContainer(nn.Linear(512*W*W,self.num_classes))
        elif self.num_classes==101:
            self.classifier = SeqToANNContainer(nn.Linear(6144,self.num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward_baseline(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x
    
    def forward_mask(self, x):
        all_mask_logits, all_gt_mask = [], []
        FLOPs = 0
        bs = x.shape[0]
        prev = torch.ones(bs) * x.shape[1]
        for i, m in enumerate(self.features.children()):
            if isinstance(m, ConvWithMask):
                import pdb;pdb.set_trace()
                x, all_mask_logits, prev, all_gt_mask, cur_flops = m(x, all_mask_logits, all_gt_mask, prev)
                FLOPs += cur_flops
            else:
                x = m(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        FLOPs = FLOPs.to(x.device)
        return x, all_mask_logits, all_gt_mask, FLOPs
    
    def forward(self, x):
        if self.baseline:
            return self.forward_baseline(x)
        else:
            return self.forward_mask(x)
    

def svgg9(num_classes=10):
    model = VGGSNN(num_classes)
    return model