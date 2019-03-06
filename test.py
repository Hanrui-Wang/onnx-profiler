import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from onnxp import *


class SEScale(nn.Module):
    def __init__(self, channels, reduction):
        super(SEScale, self).__init__()

        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers.forward(x) * x


class Shortcut(nn.Module):
    def __init__(self, stride):
        super(Shortcut, self).__init__()
        self.stride = stride

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=1, stride=self.stride)
        return torch.cat([x, x * 0.], dim=1)


class SEResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, reduction):
        super(SEResBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),

            SEScale(out_channels, reduction=reduction)
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut(stride=stride)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        return F.relu(self.layers.forward(x) + self.shortcut.forward(x), inplace=True)


class SEResNet(nn.Module):
    def __init__(self, params, dropout=None, reduction=1, **kwargs):
        super(SEResNet, self).__init__()

        layers = [
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        ]

        in_channels = 16
        for out_channels, num_blocks, strides in params:
            for stride in [strides] + [1] * (num_blocks - 1):
                layers.append(SEResBlock(in_channels, out_channels, stride=stride, reduction=reduction))
                in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_channels, 10)

        if dropout is not None:
            self.dropout = nn.Dropout(dropout, inplace=True)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features.forward(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)

        if self.dropout is not None:
            x = self.dropout.forward(x)

        x = x.view(x.size(0), -1)
        x = self.classifier.forward(x)
        return x


class SEResNet20(SEResNet):
    def __init__(self, **kwargs):
        super(SEResNet20, self).__init__([(16, 3, 1), (32, 3, 2), (64, 3, 2)], **kwargs)


if __name__ == '__main__':
    model = SEResNet20()
    inputs = torch.randn(1, 3, 32, 32)

    flops = torch_profile(model, inputs, profiler=OperationsProfiler, reduction=np.sum, verbose=True)
    params = torch_profile(model, inputs, profiler=ParametersProfile, reduction=np.sum, verbose=True)
    print(flops / 1e6, params / 1e6)
