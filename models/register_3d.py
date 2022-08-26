import torch.nn as nn
import torch
import torch.nn.functional as F
from .affine_decoder import LRB
from .encoder import Encoder
import transforms3d
import numpy as np
from math import pi, radians
from torch import cos, sin
from monai.networks.layers import AffineTransform

class Register3d(nn.Module):
    '''
    Main Model
    Encoder + Decoder => 6 DoF(3 Translation params + 3 Rotation params)
    Tanh activation func at last output so that value -1 ~ 1
    For rotation, mult pi so that -pi ~ pi
    '''
    def __init__(self):
        super(Register3d, self).__init__()
        self.encoder = Encoder()
        self.a_decoder = LRB()
        #self.affine = AffineTransform(normalized=True, align_corners=False)
        
    def compute_mat(self, x, i):
        '''
        Compose Affine Matrix without Scaling params
        '''
        tx, ty, tz = torch.tensor(x[i, 3] * pi), torch.tensor(x[i, 4] * pi), torch.tensor(x[i, 5] * pi)
        rx = torch.tensor([[1, 0, 0], [0, cos(tx), -1 * sin(tx)], [0, sin(tx), cos(tx)]])
        ry = torch.tensor([[cos(ty), 0, sin(ty)], [0, 1, 0], [-1 * sin(ty), 0, cos(ty)]])
        rz = torch.tensor([[cos(tz), -1 * sin(tz), 0], [sin(tz), cos(tz), 0], [0, 0, 1]])
        R = torch.matmul(torch.matmul(rx, ry), rz).cuda()
        return torch.cat((R, x[i, :3].unsqueeze(dim=1)), dim=1)

    def forward(self, source, target):
        x = self.encoder(source, target)
        theta = self.a_decoder(x)
        #B = source.size()[0]
        
        #l = []
        #for i in range(B):
        #    l.append(self.compute_mat(x, i))
        #theta = torch.stack(l, dim=0)
        #t = self.affine(source, theta)
        return theta #t, theta