import torch
import torch.nn.functional as F
from torch import cuda
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


class DenseNet121(torch.nn.Module):

    def __init__(self, orig_net):
        super().__init__()
        self.orig_net = orig_net
        self.final_linear = torch.nn.Linear(1024, 2)  # TODO
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.orig_net.features.conv0(x)
        x = self.orig_net.features.norm0(x)
        x = self.orig_net.features.relu0(x)
        x = self.orig_net.features.pool0(x)

        x = self.orig_net.features.denseblock1(x)
        x = self.orig_net.features.transition1(x)
        x = self.orig_net.features.denseblock2(x)
        x = self.orig_net.features.transition2(x)        
        x = self.orig_net.features.denseblock3(x)
        x = self.orig_net.features.transition3(x)        
       	x = self.orig_net.features.denseblock4(x)
        x = self.orig_net.features.norm5(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1).view(x.size(0), -1)
        x = self.final_linear(x)  # TODO
        return self.logsoftmax(x)

