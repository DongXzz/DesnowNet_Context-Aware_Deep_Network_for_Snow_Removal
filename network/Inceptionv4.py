from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import os
import sys

# __all__ = ['InceptionV4', 'inceptionv4']

# pretrained_settings = {
#     'inceptionv4': {
#         'imagenet': {
#             'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth',
#             'input_space': 'RGB',
#             'input_size': [3, 299, 299],
#             'input_range': [0, 1],
#             'mean': [0.5, 0.5, 0.5],
#             'std': [0.5, 0.5, 0.5],
#             'num_classes': 1000
#         },
#         'imagenet+background': {
#             'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth',
#             'input_space': 'RGB',
#             'input_size': [3, 299, 299],
#             'input_range': [0, 1],
#             'mean': [0.5, 0.5, 0.5],
#             'std': [0.5, 0.5, 0.5],
#             'num_classes': 1001
#         }
#     }
# }


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv = BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(80, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(80, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(32, 32, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(32, 48, kernel_size=(3,3), stride=1, padding=(1,1))
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):

    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):

    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(192, 48, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(192, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(192, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 48, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 48, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(192, 96, kernel_size=1, stride=1),
            BasicConv2d(96, 112, kernel_size=3, stride=1, padding=1),
            BasicConv2d(112, 128, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):

    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(512, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(512, 96, kernel_size=1, stride=1),
            BasicConv2d(96, 112, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(112, 128, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(512, 96, kernel_size=1, stride=1),
            BasicConv2d(96, 96, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(96, 112, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(112, 112, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(112, 128, kernel_size=(1,7), stride=1, padding=(0,3))
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(512, 64, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):

    def __init__(self):
        super(Reduction_B, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(512, 96, kernel_size=1, stride=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(512, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(128, 160, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(160, 160, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):

    def __init__(self):
        super(Inception_C, self).__init__()

        self.branch0 = BasicConv2d(768, 128, kernel_size=1, stride=1)

        self.branch1_0 = BasicConv2d(768, 192, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(192, 128, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch1_1b = BasicConv2d(192, 128, kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch2_0 = BasicConv2d(768, 192, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(192, 224, kernel_size=(3,1), stride=1, padding=(1,0))
        self.branch2_2 = BasicConv2d(224, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3a = BasicConv2d(256, 128, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3b = BasicConv2d(256, 128, kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(768, 128, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionV4(nn.Module):

    def __init__(self, input_channel, use_SE=False):
        super(InceptionV4, self).__init__()
        # # Special attributs
        # self.input_space = None
        # self.input_size = (299, 299, 3)
        # self.mean = None
        # self.std = None
        # Modules
        if use_SE:
            self.features = nn.Sequential(
                # change channel to 32
                BasicConv2d(input_channel, 16, kernel_size=3, stride=1, padding=1),
                BasicConv2d(16, 16, kernel_size=3, stride=1, padding=1),
                BasicConv2d(16, 32, kernel_size=3, stride=1, padding=1),
                Mixed_3a(),
                Mixed_4a(),
                Mixed_5a(),
                Inception_A(),
                Inception_A(),
                Inception_A(),
                Inception_A(),
                Reduction_A(), # Mixed_6a
                SELayer(512),
                Inception_B(),
                Inception_B(),
                Inception_B(),
                Inception_B(),
                Inception_B(),
                Inception_B(),
                Inception_B(),
                Reduction_B(), # Mixed_7a
                SELayer(768),
                Inception_C(),
                Inception_C(),
                Inception_C(),
                SELayer(768),
            )
        else:
            self.features = nn.Sequential(
                BasicConv2d(input_channel, 16, kernel_size=3, stride=1, padding=1),
                BasicConv2d(16, 16, kernel_size=3, stride=1, padding=1),
                BasicConv2d(16, 32, kernel_size=3, stride=1, padding=1),
                Mixed_3a(),
                Mixed_4a(),
                Mixed_5a(),
                Inception_A(),
                Inception_A(),
                Inception_A(),
                Inception_A(),
                Reduction_A(), # Mixed_6a
                Inception_B(),
                Inception_B(),
                Inception_B(),
                Inception_B(),
                Inception_B(),
                Inception_B(),
                Inception_B(),
                Reduction_B(), # Mixed_7a
                Inception_C(),
                Inception_C(),
                Inception_C()
            )
        # self.last_linear = nn.Linear(1536, num_classes)

    # def logits(self, features):
    #     #Allows image of any size to be processed
    #     adaptiveAvgPoolWidth = features.shape[2]
    #     x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
    #     x = x.view(x.size(0), -1)
    #     x = self.last_linear(x)
    #     return x

    def forward(self, input):
        x = self.features(input)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)