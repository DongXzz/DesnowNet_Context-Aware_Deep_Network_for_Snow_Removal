import torch
import torch.nn as nn
from Descriptor import Descriptor


class Pyramid_out(nn.Module):
    def __init__(self, in_channel, depth, beta, mode):
        super(Pyramid_out, self).__init__()
        assert mode in ['maxout', 'sum']
        self.mode = mode
        block = []
        for i in range(beta):
            # change to conv(1,5) and conv(5,1)
            block.append(nn.Conv2d(in_channel, depth, 2 * i + 1, 1, padding=i))
        self.conv_module = nn.ModuleList(block)
        if self.mode == 'maxout':
            self.activation = nn.PReLU(num_parameters=depth)
        else:
            self.activation = None

    def forward(self, f):
        for i, module in enumerate(self.conv_module):
            if i == 0:
                conv_result = module(f).unsqueeze(0)
            else:
                temp = module(f).unsqueeze(0)
                conv_result = torch.cat([conv_result, temp], dim=0)
        if self.mode == 'maxout':
            result_, _ = torch.max(conv_result, dim=0)
        else:
            result_ = torch.sum(conv_result, dim=0)
        if self.mode == 'maxout':
            result = self.activation(result_)
        else:
            result = torch.tanh(result_)
        return result


class R_t(nn.Module):
    # The recovery submodule (Rt) of the translucency recovery (TR) module
    def __init__(self, in_channel=385, beta=4):
        super(R_t, self).__init__()
        self.SE = Pyramid_out(in_channel, 1, beta, 'maxout')
        self.AE = Pyramid_out(in_channel, 3, beta, 'maxout')
        # self.SE = Pyramid_maxout(in_channel, 1, beta)
        # self.AE = Pyramid_maxout(in_channel, 3, beta)


    def forward(self, x, f_t, **kwargs):
        z_hat = self.SE(f_t)
        a_hat = self.AE(f_t)
        # try get a copy of z_hat than clamp as z_hat will calculate loss
        z_hat[z_hat >= 1] = 1
        z_hat[z_hat <= 0] = 0
        if 'mask' in kwargs.keys() and 'a' in kwargs.keys():
            z = kwargs['mask']
            a = kwargs['a']
        elif 'mask' in kwargs.keys():
            z = kwargs['mask']
            a = a_hat
        else:
            z = z_hat
            a = a_hat
        # yield estimated snow-free image y'
        # not let 0<a<1
        y_ = (z < 1) * (x - a_hat * z) / (1 - z + 1e-8) + (z == 1) * x
        # same for y_
        y_[y_ >= 1] = 1
        y_[y_ <= 0] = 0
        # yield feature map f_c
        if 'mask' in kwargs.keys() and 'a' in kwargs.keys():
            with torch.no_grad():
                y = (z < 1) * (x - a * z) / (1 - z + 1e-8) + (z == 1) * x
            f_c = torch.cat([y, z, a], dim=1)
        elif 'mask' in kwargs.keys():
            f_c = torch.cat([y_, z, a], dim=1)
        else:
            # Try *z to yeild a opaque snow
            f_c = torch.cat([y_, z, a], dim=1)
        return y_, f_c, z_hat, a_hat
    

class TR(nn.Module):
    # translucency recovery(TR) module
    def __init__(self, input_channel=3, beta=4, gamma=4, dp_attn=False, use_SE=False):
        super(TR, self).__init__()
        self.D_t = Descriptor(input_channel=input_channel, gamma=gamma, dp_attn=dp_attn, use_SE=use_SE)
        self.R_t = R_t(385, beta)

    def forward(self, x, **kwargs):
        f_t = self.D_t(x)
        y_, f_c, z_hat, a = self.R_t(x, f_t, **kwargs)
        return y_, f_c, z_hat, a