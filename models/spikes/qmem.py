"""
Membrane Potential Quantization
"""

import torch
from torch import Tensor

def power_quant(x:Tensor, value_s):
    shape = x.shape
    xhard = x.view(-1)
    value_s = value_s.type_as(x)
    idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]  # project to nearest quantization level
    xhard = value_s[idxs].view(shape)
    return xhard

def stats_quant(x, nbit, qmode='symm', dequantize=True):
    r"""Statistic-aware weight bining (SAWB)
    https://mlsys.org/Conferences/2019/doc/2019/168.pdf
    Compute the quantization boundary based on the stats of the distribution. 
    """
    z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}
    z = z_typical[f'{int(nbit)}bit']

    m = x.abs().mean()
    std = x.std()

    if qmode == 'symm':
        n_lv = 2 ** (nbit - 1) - 1
        alpha_w = 1/z[0] * std - z[1]/z[0] * m
    elif qmode == 'asymm':
        n_lv = 2 ** (nbit - 1) - 1
        alpha_w = 2*m
    else:
        raise NotImplemented

    x = x.clamp(-alpha_w.item(), alpha_w.item())
    scale = n_lv / alpha_w
    
    xq = x.mul(scale).round()
    if len(xq.unique()) > 2**nbit:
        xq = xq.clamp(-2**nbit//2, 2**nbit//2-1)
    
    if dequantize:
        xq = xq.div(scale)
    return xq, scale

def pqmem(mem:Tensor, levels:Tensor, neg=-1.0, thresh=1.0):
    mem = mem.clamp(min=neg, max=thresh)
    memq = power_quant(mem, levels)
    return memq

def sigmoid(x:Tensor, T:float=5.0, s=0.5):
    e = T * (x + s)
    return torch.sigmoid(e)

def dsigmoid(x:Tensor, T:float=5.0, s=0.5):
    sig = sigmoid(x, T, s)
    return T * (1-sig) * sig

class TernMem(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, levels):
        out = pqmem(inputs, levels, neg=levels.min().item())
        ctx.save_for_backward(inputs)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors[0]
        grad_input = grad_output.clone()

        sg = dsigmoid(inputs, T=2.0, s=0.5) + dsigmoid(inputs, T=2.0, s=-0.5)
        grad_input = grad_input * sg
        return grad_input, None

class QMem(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, levels, interval):
        out = pqmem(inputs, levels, neg=levels.min().item())
        ctx.save_for_backward(inputs, levels, interval)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, levels, interval = ctx.saved_tensors
        grad_input = grad_output.clone()

        sg = 0.0
        for i, l in enumerate(levels):
            shift = interval / 2
            
            if l != 0:
                s = l + shift if l < 0 else l - shift
                sg += dsigmoid(inputs, T=2.0, s=-s)
        
        grad_input = grad_input * sg
        return grad_input, None, None
