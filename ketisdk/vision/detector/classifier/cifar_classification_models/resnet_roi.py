from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch.nn as nn
import math
from .resnet import ResNet
import torch.nn.functional as F
import torch


__all__ = ['resnet_roi']

class ResNetRoi(ResNet):
    def __init__(self, depth, input_shape=(32, 32, 3), num_classes=1000, block_name='BasicBlock'):
        super().__init__(depth, input_shape=input_shape, num_classes=num_classes, block_name=block_name)
        self.scaling = nn.Upsample(size=input_shape[:2],mode='bilinear')

    def forward(self, x):
        ims ,boxes, inds = x
        roi_concat = []
        for i, box in zip(inds, boxes):
            left, top, right, bottom = box
            roi = ims[i][:,top:bottom, left:right]
            roi_scale = self.scaling(roi.view(1, *roi.size()))[0,:,:,:]
            roi_concat.append(roi_scale)
        roi_concat = torch.stack(roi_concat)
        predicted = super().forward(roi_concat)
        return F.softmax(predicted, dim=1)

def resnet_roi(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNetRoi(**kwargs)
