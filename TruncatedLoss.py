import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from parser import opt

class TruncatedLoss(nn.Module):

    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.trainset_size = trainset_size
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)

    def forward(self, logits, targets, indexes):

        p = logits
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        loss = ((1 - (Yg ** self.q)) / self.q) * self.weight[indexes] - ((1 - (self.k ** self.q)) / self.q)*self.weight[indexes]
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = logits
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1 - (Yg ** self.q)) / self.q)
        Lqk = np.repeat(((1 - (self.k ** self.q)) / self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1).cuda(opt.cuda_device)

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.FloatTensor).cuda(opt.cuda_device)

    def forward_nomean(self, logits, targets, indexes):

        p = logits
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        loss = ((1 - (Yg ** self.q)) / self.q) * self.weight[indexes] - ((1 - (self.k ** self.q)) / self.q)*self.weight[indexes]

        return loss

    def clear_weight(self):
        self.weight = torch.nn.Parameter(data=torch.ones(self.trainset_size, 1).cuda(opt.cuda_device), requires_grad=False)


































