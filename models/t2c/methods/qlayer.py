"""
Customized quantization layers and modules

Example method:
SAWB-PACT: Accurate and Efficient 2-bit Quantized Neural Networks
RCF: Additive Power-of-Two Quantization: An Efficient Non-uniform Discretization For Neural Networks
"""
import torch
import torch.nn.functional as F
from torch import Tensor
from .qexample import STE
from .base import QBaseConv2d, QBase, QBaseLinear

class SAWB(QBase):
    def __init__(self, nbit: int, train_flag: bool = True, qmode:str="symm"):
        super(SAWB, self).__init__(nbit, train_flag)
        self.register_buffer("alpha", torch.tensor(1.0))
        self.register_buffer("scale", torch.tensor(1.0))
        self.qmode = qmode

        # sawb
        z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}
        self.z = z_typical[f'{int(nbit)}bit']

    def q(self, input:Tensor):
        """
        Quantization method
        """
        m = input.abs().mean()
        std = input.std()

        if self.qmode == 'symm':
            n_lv = 2 ** (self.nbit - 1) - 1
            self.alpha.data = 1/self.z[0] * std - self.z[1]/self.z[0] * m
        elif self.qmode == 'asymm':
            n_lv = 2 ** (self.nbit - 1) - 1
            self.alpha.data = 2*m
        else:
            raise NotImplemented
    
        self.scale.data = n_lv / self.alpha
        
        if not self.train_flag:
            xq = input.clamp(-self.alpha.item(), self.alpha.item())
            xq = xq.mul(self.scale).round()
            if len(xq.unique()) > 2**self.nbit:
                xq = xq.clamp(-2**self.nbit//2, 2**self.nbit//2-1)
            
            if self.dequantize:
                xq = xq.div(self.scale)
        else:
            xq = input
        return xq

    def trainFunc(self, input:Tensor):
        input = input.clamp(-self.alpha.item(), self.alpha.item())
        # get scaler
        _ = self.q(input)
        # quantization-aware-training
        out = STE.apply(input, self.scale.data)
        return out
    
    def evalFunc(self, input: Tensor):
        out = self.q(input)
        return out
        
class TernFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        tFactor = 0.05
        
        max_w = input.abs().max()
        th = tFactor*max_w #threshold
        output = input.clone().zero_()
        W = input[input.ge(th)+input.le(-th)].abs().mean()
        output[input.ge(th)] = W
        output[input.lt(-th)] = -W

        return output
    @staticmethod
    def backward(ctx, grad_output):
        # saved tensors - tuple of tensors with one element
        grad_input = grad_output.clone()
        return grad_input

class TernW(QBase):
    def __init__(self, nbit: int=2, train_flag: bool = True):
        super().__init__(nbit, train_flag)
        self.tFactor = 0.05
    
    def trainFunc(self, input: Tensor):
        out = TernFunc.apply(input)
        return out


class QConv2d(QBaseConv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, wbit: int = 32, abit: int = 32, train_flag=True):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, wbit, abit, train_flag)
        # layer index
        self.layer_idx = 0
        
        # quantizers
        if wbit < 32:
            if wbit in [4, 8]:
                self.wq = SAWB(self.wbit, train_flag=True, qmode="asymm")
            elif wbit in [2]:
                self.wq = TernW(train_flag=True)
        

    def forward(self, input:Tensor):
        wq = self.wq(self.weight)
        y = F.conv2d(input, wq, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y

class QLinear(QBaseLinear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, wbit: int = 32, abit: int = 32, train_flag=True):
        super(QLinear, self).__init__(in_features, out_features, bias, wbit, abit, train_flag)

        # quantizers
        if wbit < 32:
            if wbit in [4, 8]:
                self.wq = SAWB(self.wbit, train_flag=True, qmode="asymm")

    def trainFunc(self, input):
        return super().trainFunc(input)