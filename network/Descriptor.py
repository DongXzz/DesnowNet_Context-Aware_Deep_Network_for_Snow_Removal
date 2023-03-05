import torch
import torch.nn as nn
from Inceptionv4 import InceptionV4


class DP(nn.Module):
    # dilation pyramid
    def __init__(self, in_channel=1536, depth=96, gamma=4):
        super(DP, self).__init__()
        self.gamma = gamma
        block = []
        for i in range(gamma + 1):
            block.append(nn.Conv2d(in_channel, depth, 3, 1, padding=2 ** i, dilation=2 ** i))
        self.block = nn.ModuleList(block)

    def forward(self, feature):
        for i, block in enumerate(self.block):
            if i == 0:
                output = block(feature)
            else:
                output = torch.cat([output, block(feature)], dim=1)
        return output

class DP_attn(nn.Module):
    # dilation pyramid
    def __init__(self, in_channel=768, out_channel=385, num_heads=5):
        super(DP_attn, self).__init__()
        self.num_heads = num_heads
        self.query_conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=1, stride=1,
                              bias=False)
        self.key_conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=1, stride=1,
                              bias=False)
        self.value_conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=1, stride=1,
                              bias=False)
        
        self.multihead_attn = nn.MultiheadAttention(out_channel, num_heads)

    def forward(self, feature):
        n, c, w, h = feature.shape
        query_ = self.query_conv(feature)
        query = query_.view(n, -1, w*h).permute((2, 0, 1))
        key_ = self.query_conv(feature)
        key = key_.view(n, -1, w*h).permute((2, 0, 1))
        value_ = self.query_conv(feature)
        value = value_.view(n, -1, w*h).permute((2, 0, 1))
        attn_output_, attn_output_weights = self.multihead_attn(query, key, value)
        attn_output = attn_output_.permute((1, 2, 0)).view(n, -1, w, h)
        return attn_output

class Descriptor(nn.Module):
    def __init__(self, input_channel=3, factor=1, gamma=4, dp_attn=False, use_SE=False):
        super(Descriptor, self).__init__()
        self.backbone = InceptionV4(input_channel, factor=factor, use_SE=use_SE)
        if dp_attn:
            self.DP = DP_attn()
        else:
            self.DP = DP(in_channel=1536//factor, gamma=gamma)

    def forward(self, img):
        feature = self.backbone(img)
        f = self.DP(feature)
        return f


if __name__ == '__main__':
    device = 'cpu'
    Descriptor_1 = Descriptor().to(device)
    img = torch.zeros([1, 3, 200, 200]).to(device)
    f = Descriptor_1(img)
    f.mean().backward()
    print("finished")
