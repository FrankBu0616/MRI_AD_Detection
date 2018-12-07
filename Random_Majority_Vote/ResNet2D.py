import torch
import torch.nn.functional as F
from torch import cuda
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


class BinaryResnet(torch.nn.Module):

    def __init__(self, orig_resnet):
        super().__init__()
        self.orig_resnet = orig_resnet
        self.final_linear = torch.nn.Linear(512, 2)  # TODO
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.orig_resnet.conv1(x)
        x = self.orig_resnet.bn1(x)
        x = self.orig_resnet.relu(x)
        x = self.orig_resnet.maxpool(x)

        x = self.orig_resnet.layer1(x)
        x = self.orig_resnet.layer2(x)
        x = self.orig_resnet.layer3(x)
        x = self.orig_resnet.layer4(x)
        x = self.orig_resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.final_linear(x)  # TODO

        return self.logsoftmax(x)

