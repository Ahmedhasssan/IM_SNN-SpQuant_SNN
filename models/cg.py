import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SparseGreaterThan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.Tensor.float(torch.gt(input, torch.zeros_like(input)))
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<torch.zeros_like(input)] = 0
        return grad_input, None

class GreaterThan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.Tensor.float(torch.gt(input, torch.zeros_like(input)))
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class GreaterThanSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.Tensor.float(torch.gt(input, torch.zeros_like(input)))
    @staticmethod
    def backward(ctx, grad_output):
        alpha = 1.0
        input, = ctx.saved_tensors
        grad_input = alpha * grad_output.clone()
        grad_input[input < (-0.5/alpha)*torch.ones_like(input)] = 0
        grad_input[input > (0.5/alpha)*torch.ones_like(input)] = 0
        return grad_input, None

class CGConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,
                 padding_mode='zeros', sparse_bp=False, use_group=False,
                 shuffle=False, p=4, th=-6.0, alpha=2.0):
        super(CGConv2d, self).__init__(in_channels, out_channels, 
                                       kernel_size, stride, 
                                       padding, dilation, groups, 
                                       bias, padding_mode)
        self.gt = SparseGreaterThan.apply if sparse_bp else GreaterThan.apply
        self.gtSTE = GreaterThanSTE.apply
        self.th = th
        self.alpha = alpha
        self.p = p
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.shuffle = shuffle

        """
        initialize the mask for the weights
        """
        in_chunk_size = int(in_channels/self.p)
        out_chunk_size = int(out_channels/self.p)
        
        mask = torch.zeros(out_channels, in_channels, kernel_size, kernel_size)
        if use_group:
            for idx in range(self.p):
                mask[idx*out_chunk_size:(idx+1)*out_chunk_size, idx*in_chunk_size:(idx+1)*in_chunk_size] = torch.ones(out_chunk_size, in_chunk_size, kernel_size, kernel_size)
        else:
            mask[:, 0:in_chunk_size] = torch.ones(out_channels, in_chunk_size, kernel_size, kernel_size)
        self.mask = nn.Parameter(mask, requires_grad=False)

        """ 
        initialize the threshold with th
        """
        self.threshold = nn.Parameter(self.th * torch.ones(1, out_channels, 1, 1))

        """ number of output features """
        self.num_out = 0
        """ n!umber of output features computed using all input channels """
        self.num_full = 0

    def forward(self, input):
        """ 
        1. mask the weight tensor
        2. compute Yp
        3. generate gating decision d
        """
        if self.shuffle:
            input = channel_shuffle(input, self.p) 
        Yp = F.conv2d(input, self.weight * self.mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
        """ Calculate the gating decison d """
        d = self.gt(torch.sigmoid(self.alpha*(self.bn(Yp)-self.threshold)) - 0.5 * torch.ones_like(Yp))
        """ update report """
        self.num_out = d.numel()
        self.num_full = d[d>0].numel()
        import pdb;pdb.set_trace()
        """ perform full convolution """
        Y = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        """ combine outputs """
        return Y * d + Yp * (torch.ones_like(d) - d)