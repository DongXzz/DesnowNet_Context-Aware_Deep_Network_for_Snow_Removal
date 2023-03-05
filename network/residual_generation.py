import torch.nn as nn
from Descriptor import Descriptor
from translucency_recovery import Pyramid_out

class RG(nn.Module):
    # the residual generation (RG) module
    def __init__(self, input_channel=7, beta=4, gamma=4):
        super(RG, self).__init__()
        self.D_r = Descriptor(input_channel, gamma)
        self.R_r = Pyramid_out(480, 3, beta, 'sum')

    def forward(self, f_c):
        f_r = self.D_r(f_c)
        r = self.R_r(f_r)
        return r