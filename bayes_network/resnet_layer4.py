'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .BayesianConvs import BayesianConv2D
from .BatchNorm import BayesianBatchNorm2d
from torch.autograd import Variable


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x,sample=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bayes_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes,args, stride=1):
        super(Bayes_BasicBlock, self).__init__()
        self.conv1 = BayesianConv2D(in_planes, planes, 3, args,stride=stride, padding=1, use_bias=True)
        self.bn1 = BayesianBatchNorm2d(planes, args)
        self.conv2 = BayesianConv2D(planes, planes, 3, args,stride=1, padding=1, use_bias=True)
        self.bn2 = BayesianBatchNorm2d(planes, args)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                BayesianConv2D(in_planes, self.expansion*planes, 1, args,stride=stride, use_bias=True),
                BayesianBatchNorm2d(self.expansion*planes,args)
            )

    def forward(self, x,sample=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, Bayesblock, num_blocks, args, num_classes=7):
        super(ResNet, self).__init__()

        self.in_planes = 64
        self.args = args

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_Bayeslayer(Bayesblock, 512, num_blocks[3], stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.avgpool = nn.AvgPool2d(2, stride=2)
        self.linear = nn.Linear(512*2*2, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_Bayeslayer(self, Bayesblock, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Bayesblock(self.in_planes, planes,self.args, stride))
            self.in_planes = planes * Bayesblock.expansion
        return nn.Sequential(*layers)

    def forward(self, x,sample=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # out = F.dropout(out, p=0.5, training=self.training)
        out=self.linear(out)
        return out


def Net(args):
    # return ResNet(BasicBlock, [3, 4, 6, 3], args)#[2,2,2,2]
    return ResNet(BasicBlock, Bayes_BasicBlock,[2,2,2,2], args)
