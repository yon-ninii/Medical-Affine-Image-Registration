import torch.nn.functional as F
import torch.nn as nn
import torch
import math
from math import sin, cos, radians, pi
import numpy as np

class LRB(nn.Module):
    '''
    Decoder Module
    1 adaptive average pooling
    2 FC layer
    6 DoF output
    '''
    def __init__(self):
        super(LRB, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool3d((12, 12, 12))
        self.fc1 = nn.Linear(501120, 960)
        self.fc2 = nn.Linear(960, 6)

    def forward(self, inputs):
        x = self.pool1(inputs)
        B = x.size()[0]
        x = x.view(B, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.tanh(x)
        return x #.view(B, 3, 4)