import copy
import os
# we already use worker threads, limit each process to 1 thread!
from functools import partial
from numbers import Number
from typing import Optional, Sequence

from trw.basic_typing import Stride, KernelSize
from trw.layers import LayerConfig, BlockConv, BlockConvNormActivation, default_layer_config

import trw
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from trw.layers.layer_config import PoolType

torch.set_num_threads(1)

import logging

logging.basicConfig(filename='log.log', filemode='w', level=logging.DEBUG)








class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18():
    return PreActResNet(PreActBlock, [2,2,2,2])

def PreActResNet18v2():
    return PreActResNetV2(
        dimensionality=2,
        input_channels=3,
        output_channels=10,
        block=BlockResPreAct,
        num_blocks=[2, 2, 2, 2])

def PreActResNet34():
    return PreActResNet(PreActBlock, [3,4,6,3])

def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3,4,6,3])

def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = trw.layers.PreActResNet18()

    def forward(self, batch):
        x = batch['images']
        x = self.net(x)
        return {
            'softmax': trw.train.OutputClassification(x, batch['targets'], classes_name='targets')
        }


def create_model():
    model = Net()
    return model


if __name__ == '__main__':
    # configure and run the training/evaluation
    num_epochs = 200
    options = trw.train.Options(num_epochs=num_epochs)
    trainer = trw.train.TrainerV2(
        callbacks_post_training=None,
        callbacks_pre_training=None,
    )

    mean = np.asarray([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std = np.asarray([0.2023, 0.1994, 0.2010], dtype=np.float32)

    transform_train = [
        trw.transforms.TransformRandomCropPad(padding=[0, 4, 4], mode='constant'),
        trw.transforms.TransformRandomFlip(axis=3),
        trw.transforms.TransformRandomCutout(cutout_size=(3, 16, 16), probability=0.2),
        trw.transforms.TransformNormalizeIntensity(mean=mean, std=std)
    ]

    transform_valid = [
        trw.transforms.TransformNormalizeIntensity(mean=mean, std=std)
    ]

    datasets = trw.datasets.create_cifar10_dataset(
        transform_train=transform_train,
        transform_valid=transform_valid,
        nb_workers=4,
        batch_size=128,
        data_processing_batch_size=64,
    )

    scheduler_fn = lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    optimizer_fn_1 = lambda datasets, model: trw.train.create_sgd_optimizers_fn(
        datasets,
        model,
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=False,
        scheduler_fn=scheduler_fn
    )

    results = trainer.fit(
        options,
        datasets=datasets,
        log_path='cifar10_resnet',
        model=create_model(),
        optimizers_fn=optimizer_fn_1
    )


