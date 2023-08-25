import torch.nn as nn
from ..spikes import SeqToANNContainer, QLIFSpike
from ..t2c.methods import QConv2d
from ..cg import CGConv2d
from models.layer import ConvWithMask
import torch

class SConv(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding, pool=False, wbit=32, tau=0.5):
        super(SConv, self).__init__()
        counter = 0
        self.flops_conv = 0
        self.actual_flops_conv = 0
        self.active_channels = 0
        self.total_channels = 0
        if wbit < 32:
            self.fwd = SeqToANNContainer(
                QConv2d(in_plane, out_plane, kernel_size, stride, padding, wbit=wbit, abit=32),
                nn.BatchNorm2d(out_plane)
            )
        else:
            conv = nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding)
            batchnorm = nn.BatchNorm2d(out_plane)
            act = nn.ReLU()
            self.conv_mask =ConvWithMask(conv, batchnorm, act, target = -1, nclass=10, ratio=0.85, layerid = counter, do_softmax = 1, mode='decoupled', gt_type='mass').to('cuda')
            self.fwd = SeqToANNContainer(
                nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
                nn.BatchNorm2d(out_plane)
            )
            counter += 1
        self.act = QLIFSpike(tau=tau)

        if pool:
            self.pool = SeqToANNContainer(nn.AvgPool2d(2))
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        all_mask_logits, all_gt_mask = [], []
        bs = x.shape[0]
        prev = torch.ones(bs) * x.shape[2]
        input_l = x
        out, binary_mask ,all_mask_logits, active_cur, all_gt_mask, FLOPs, act_FLOPs = self.conv_mask(x, all_mask_logits, all_gt_mask, prev)
        x = self.fwd(x)
        x = self.pool(x)
        x = self.act(x, binary_mask, input_l)
        self.flops_conv = FLOPs.mean()
        self.active_channels += active_cur.mean()
        self.total_channels += all_mask_logits[0].size(1)
        self.actual_flops_conv = act_FLOPs.mean()
        #print(len(binary_mask[binary_mask>0]))
        return x
    
class SConvDW(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding, pool=False, wbit=32, tau=0.5):
        super(SConvDW, self).__init__()
        if wbit < 32:
            self.dw = SeqToANNContainer(
                QConv2d(in_plane, in_plane, kernel_size, stride, padding, groups=in_plane, wbit=wbit, abit=32),
                nn.BatchNorm2d(in_plane)
            )
            self.pw = SeqToANNContainer(
                QConv2d(in_plane, out_plane, 1, stride, padding=0, wbit=wbit, abit=32),
                nn.BatchNorm2d(out_plane)
            )
        else:
            self.dw = SeqToANNContainer(
                nn.Conv2d(in_plane, in_plane, kernel_size, stride, padding, groups=in_plane),
                nn.BatchNorm2d(in_plane)
            )
            self.pw = SeqToANNContainer(
                nn.Conv2d(in_plane, out_plane, 1, stride, padding=0),
                nn.BatchNorm2d(out_plane)
            )
        self.act1 = QLIFSpike(tau=tau)
        self.act2 = QLIFSpike(tau=tau)
        
        if pool:
            self.pool = SeqToANNContainer(nn.AvgPool2d(2))
            # self.pool = SeqToANNContainer(nn.MaxPool2d(2))
        else:
            self.pool = nn.Identity()

    def forward(self,x):
        x = self.dw(x)
        x = self.act1(x)
        x = self.pw(x)
        x = self.pool(x)
        x = self.act2(x)
        return x