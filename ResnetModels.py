import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import torchvision

class ResNet101(nn.Module):
    def __init__(self,num_classes,pretrained,transfer_learn=False, groups=3):
        super(ResNet101,self).__init__()
        self.resnet101 = torchvision.models.resnet101(pretrained=pretrained)
        # If transfer learning, freeze gradients flow
        if transfer_learn==True:
                for param in self.resnet101.parameters():
                    param.requires_grad = False
        self.classify = nn.Sequential(nn.Linear(512*4 ,int(num_classes)),nn.Sigmoid())
    def forward(self,y):
        y = self.resnet101.conv1(y)
        y = self.resnet101.bn1(y)        
        y = self.resnet101.relu(y)
        y = self.resnet101.maxpool(y)

        y = self.resnet101.layer1(y)
        y = self.resnet101.layer2(y)
        y = self.resnet101.layer3(y)
        y = self.resnet101.layer4(y)

        y = self.resnet101.avgpool(y)
        y = torch.flatten(y,1)
        y = self.classify(y)
        return y