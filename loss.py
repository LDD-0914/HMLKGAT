import torch as t
from torch import nn
import numpy as np

class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, target, prediction,drug_lap, mic_lap, alpha1, alpha2, sizes):

        loss_ls = t.norm((target - prediction), p='fro') ** 2
        drug_reg = t.trace(t.mm(t.mm(alpha1.T, drug_lap), alpha1))
        drug_reg2 = t.norm(alpha1, p='fro') ** 2
        mic_reg = t.trace(t.mm(t.mm(alpha2.T, mic_lap), alpha2))
        mic_reg2 = t.norm(alpha2, p='fro') ** 2
        # graph_reg = sizes.lambda1 * drug_reg + sizes.lambda2 * mic_reg+sizes.lambda3*drug_reg2+sizes.lambda4*mic_reg2
        graph_reg =  sizes.lambda3 * drug_reg2 + sizes.lambda4 * mic_reg2
        loss_sum = loss_ls + graph_reg

        return loss_sum.sum()

