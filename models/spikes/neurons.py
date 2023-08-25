import torch
import torch.nn as nn
from .qmem import power_quant, QMem, TernMem
from utils import AverageMeter
import csv
import statistics
import numpy as np
from .mask import *

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply

        self.thresh = thresh
        self.tau = tau
        self.gama = gama

        # meters
        self.sr = AverageMeter()

    def pqmem(self, mem:torch.Tensor):
        mem = mem.clamp(min=self.neg, max=self.thresh)
        memq = power_quant(mem, self.levels)
        return memq
    
    def qmem(self, mem:torch.Tensor):
        mem = mem.clamp(min=self.neg, max=self.thresh)
        memq = mem.mul(self.scale).round().div(self.scale)
        return memq

    def fire_rate(self, mem:torch.Tensor, spike:torch.Tensor):
        mmask = mem.eq(0.).float()
        res = torch.bitwise_and(mmask.int(), spike.int())

        # spike rate 
        sr = res[res.eq(1.0)].numel() / spike.numel()
        return sr, mmask.sum() / mmask.numel()
    
    def save_mem(self, mem:torch.Tensor, t:int):
        # save the membrane potential
        pot = mem.detach().cpu()
        torch.save(pot, f"./mem_pot/neuron{self.neuron_idx}_t{t}.pt")

    def save_y(self, y:torch.Tensor):
        pot = y.detach().cpu()
        torch.save(pot, f"./conv_out/conv_out{self.neuron_idx}.pt")

    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[1]

        for t in range(T):
            tmp = mem * self.tau
            mem = tmp + x[:, t, ...]
            spike = self.act(mem - self.thresh, self.gama)
    
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)

class QLIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0, lb=-3.0):
        super(QLIFSpike, self).__init__()
        self.act = ZIF.apply
        self.qfunc = QMem.apply

        self.thresh = thresh
        self.tau = tau
        self.gama = gama

        # membrane potential quantization
        self.lb = lb
        ub = thresh + 1
        qrange = ub - lb
        self.interval = torch.tensor(1.0)
        self.levels = torch.tensor([lb + self.interval*i for i in range(int(qrange / self.interval))])

        # meters
        self.sr = 0
        self.nonzero_channels = 0
        self.pixels_dist = 0
        #self.sp_mask = Mask_s.apply
        self.flops_mem = 0
        self.actual_flops_mem = 0
        self.sparsity = 0
        self.flops_reduction = 0

    def forward(self, x, mask, conv_input):
        mask = mask.unsqueeze_(1)
        #mask = mask.repeat(1,x.size(1),1,1,1)
        mask = mask.reshape(x.size(0),x.size(1),x.size(2),1,1)
        x = x*mask
        # active_cur = mask.mean(dim=(3,4)).sum(1).detach().cpu()
        # active_cur = active_cur.mean(0)
        active_cur = mask.reshape(mask.size(0)*mask.size(1)*mask.size(2))
        active_cur = len(torch.nonzero(active_cur))
        total_channel = len(mask.reshape(mask.size(0)*mask.size(1)*mask.size(2)))
        pixels = x.size(3)*x.size(4)

        mem = 0
        sr_ind = []
        spike_pot = []

        pos_pixels = []
        neg_pixels = []
        channels = []
        final_dict = {}
        flops = 0
        sparse = []
        original_flops = 0
        #T = x.shape[1]
        T = 10
        sp_mask = Mask_s(x.size(3),x.size(4),x.size(2),2,2)
        s=0
        for t in range(T):
            # if t==9:
            #     for m in range(mem.size(1)):
            #         pos_pixels.append(np.sum(mem[:,m,...].cpu().detach().numpy()>=-0.8))
            #         neg_pixels.append(np.sum(mem[:,m,...].cpu().detach().numpy()<-0.8))
            #         channels.append(m)
            #     final_dict.update({"channels":channels,"pos_pixels":pos_pixels, "neg_pixels":neg_pixels})
            tmp = mem * self.tau
            mem = tmp + x[:, t, ...]
            sparsity_actual = torch.sum(mem==0).item()/(torch.nonzero(mem).size(0)+torch.sum(mem==0).item())
            # mask_s, norm, norm_t = sp_mask(mem)
            # mem = mem*mask_s
            sparsity = torch.sum(mem==0).item()/(torch.nonzero(mem).size(0)+torch.sum(mem==0).item())
            flops += x.size(0)*((active_cur)*pixels*(1-sparsity))
            original_flops += x.size(0)*(total_channel)*pixels
            sparse.append(sparsity)
            spike = self.act(mem - self.thresh, self.gama)
            # s =spike.sum([2,3])/(spike.size(2)*spike.size(3))
            # s=torch.mean(s, dim=0)

            mem = (1 - spike) * mem
            # q mem
            mem = self.qfunc(mem, self.levels, self.interval)
            spike_pot.append(spike)

        # sr_ind.append(s.tolist())
        # self.sr = sr_ind
        #self.pixels_dist = final_dict
        self.flops_mem = flops
        self.actual_flops_mem = original_flops
        # self.sparsity = sum(sparse) / len(sparse)
        # self.flops_reduction = (1-flops/original_flops)*100
        return torch.stack(spike_pot, dim=1)

    def extra_repr(self) -> str:
        return super().extra_repr() + ", lb={:.2f}, interval={:.2f}".format(self.lb, self.interval.item())
    

# class QLIFSpike(nn.Module):
#     def __init__(self, thresh=1.0, tau=0.5, gama=1.0, lb=-3.0):
#         super(QLIFSpike, self).__init__()
#         self.act = ZIF.apply
#         self.qfunc = QMem.apply

#         self.thresh = thresh
#         self.tau = tau
#         self.gama = gama

#         # membrane potential quantization
#         self.lb = lb
#         ub = thresh + 1
#         qrange = ub - lb
#         self.interval = torch.tensor(1.0)
#         self.levels = torch.tensor([lb + self.interval*i for i in range(int(qrange / self.interval))])

#         # meters
#         self.sr = 0
#         self.total_channels = 0
#         self.nonzero_channels = 0
#         self.pixels_dist = 0

#     def forward(self, x, mask):
#         mem = 0
#         #sr_ind = torch.empty(1,64).cuda()
#         sr_ind = []
#         spike_pot = []
#         mmask = 1
#         count = 0
#         #T = x.shape[1]
#         pos_pixels = []
#         neg_pixels = []
#         channels = []
#         final_dict = {}
#         T = 10
#         s=0
#         for t in range(T):
#             # if t>0:
#             #     mask = 0
#             #     mmask = torch.zeros(x.size(0), 1,x.size(3),x.size(4))
#             #     for m in range(mem.size(1)):
#             #         me =mem[:,m, ...].sum([1,2])/(mem[:,m, ...].size(1)*mem[:,m, ...].size(2))
#             #         me = torch.mean(me, dim=0)
#             #         if me >= -0.8:
#             #             mask = torch.ones(mem.size(0), 1,mem.size(2),mem.size(3))
#             #             mmask = torch.cat((mmask,mask), dim=1)
#             #         elif me<-0.8:
#             #             mask = torch.zeros(mem.size(0), 1,mem.size(2),mem.size(3))
#             #             mmask = torch.cat((mmask,mask), dim=1)
#             #         #mmask = me.eq(-1).float()
#             #     mmask = mmask [:,:-1,...]
#             if t==9:
#                 for m in range(mem.size(1)):
#                     pos_pixels.append(np.sum(mem[:,m,...].cpu().detach().numpy()>=-0.8))
#                     neg_pixels.append(np.sum(mem[:,m,...].cpu().detach().numpy()<-0.8))
#                     channels.append(m)
#                 final_dict.update({"channels":channels,"pos_pixels":pos_pixels, "neg_pixels":neg_pixels})
#             tmp = mem * self.tau
#             mem = tmp + x[:, t, ...]
#             spike = self.act(mem - self.thresh, self.gama)
#             # if t>0:
#             #     spike = spike.mul(mmask.cuda())
#             # #### Spike Pruning Analysis######
#             # s =spike.sum([2,3])/(spike.size(2)*spike.size(3))
#             # s=torch.mean(s, dim=0)
#             # #sr_ind.append(s.tolist())
#             # ############################
#             ### Conv input Analysis######
#             # s =conv_input.sum([3,4])/(conv_input.size(3)*conv_input.size(4))
#             # s=torch.mean(s, dim=0)
#             # s=torch.mean(s, dim=0)
#             ###########################
#             mem = (1 - spike) * mem
#             # q mem
#             mem = self.qfunc(mem, self.levels, self.interval)
#             spike_pot.append(spike)
#             # self.nonzero_channels += np.count_nonzero(spike.cpu().detach().numpy())
#             # self.total_channels += np.size(spike.cpu().detach().numpy())
#         # sr_ind.append(s.tolist())
#         # self.sr = sr_ind
#         # print(self.nonzero_channels)
#         # print(self.total_channels)
#         # import pdb;pdb.set_trace()
#         self.pixels_dist = final_dict
#         return torch.stack(spike_pot, dim=1)

#     def extra_repr(self) -> str:
#         return super().extra_repr() + ", lb={:.2f}, interval={:.2f}".format(self.lb, self.interval.item())