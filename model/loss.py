import torch
import torch.nn as nn

#定义FocalLoss损失函数
class FocalLoss(nn.Module):

    def __init__(self,gamma):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self,input,target):
        logp = self.ce(input,target)
        p = torch.exp(-logp)
        loss = (1-p)**self.gamma*logp
        return loss.mean()
