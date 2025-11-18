import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import pandas as pd
import os
import dataset
import matplotlib.pyplot as plt
import utils
import json


class MultiMagCNN(nn.Module):
    def __init__(self,num_classes = 6, num_mags_modes = 8, dropout=0.5):
        super(MultiMagCNN, self).__init__()
        self.num_mags_modes =num_mags_modes
        
        
        self.convultions = nn.Sequential(*list(models.resnet34(weights='DEFAULT').children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        num_ftrs = 512*2
        self.pool_bn = nn.BatchNorm1d(num_ftrs)
        self.linseq = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs)
            )
        feat_in = (num_ftrs*num_mags_modes)
        self.BN1 = torch.nn.BatchNorm1d(feat_in)   
        self.fc_out = nn.Linear(feat_in, num_classes)        
        self.do = torch.nn.Dropout(p=dropout)
        
    def forward(self, x):
        output = []
        for i in range(self.num_mags_modes):
            xi = x[:,i,:,:,:]
            out1 = self.convultions(xi)
            gmp = self.max_pool(out1)
            gap = self.avg_pool(out1)
            out2 = torch.cat([gmp,gap],1)
            out3 = torch.squeeze(out2)
            out4 = self.pool_bn(out3)
            out5 = self.linseq(out4)
            output.append(F.relu(out5,inplace=True))
        output = torch.cat(output,1)
        output1 = self.BN1(output)
        output2 = self.do(output1)
        return self.fc_out(output2)

class ProtoNet(nn.Module):
    def __init__(self, num_mags_modes = 6, dropout=0.0, embeddingNum =64):
        super(ProtoNet, self).__init__()
        self.num_mags_modes =num_mags_modes
        self.convultions = nn.Sequential(*list(models.resnet34(weights='DEFAULT').children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        num_ftrs = 512*2
        self.pool_bn = nn.BatchNorm1d(num_ftrs)
        self.linseq = nn.Sequential(
            nn.Linear(num_ftrs, embeddingNum)
            )
        feat_in = (embeddingNum*num_mags_modes)
        self.BN1 = torch.nn.BatchNorm1d(feat_in)       
        self.do = torch.nn.Dropout(p=dropout)
        
    def forward(self, x):
        output = []
        for i in range(self.num_mags_modes):
            xi = x[:,i,:,:,:]
            out1 = self.convultions(xi)
            gmp = self.max_pool(out1)
            gap = self.avg_pool(out1)
            out2 = torch.cat([gmp,gap],1)
            out3 = torch.squeeze(out2)
            out4 = self.pool_bn(out3)
            out5 = self.linseq(out4)
            output.append(F.relu(out5,inplace=True))
        output = torch.cat(output,1)
        output1 = self.BN1(output)
        output2 = self.do(output1)
        return output2