import torch.nn.functional as F
import torch.nn as nn


class Transformer3d(nn.Module):
    def __init__(self):
        super(Transformer3d, self).__init__()

    def forward(self, source, affine_grid):
        #x = F.grid_sample(source, grad_grid)
        #if affine_grid is not None:
        x = F.grid_sample(source, affine_grid, align_corners = True)
        return x
