import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import random
import csv
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt



class MultiImage(Dataset):
    def __init__(self, df, ROUTES, magnification, mode, datatype):
        self.datatype = datatype
        self.magnification = magnification
        self.mode = mode
        self.df = df
        self.transf = self._get_transforms(self.datatype)
        self.label_index = {}
        for i,r in enumerate(ROUTES):
            self.label_index[r]=i
        
    def __len__(self):
        return len(self.df['Set'].unique())

    def __getitem__(self, idx):
        names, label, directory = self.get_file_names_labels(idx)
        images = []
        for n in names:
            im = Image.open(os.path.join(directory,n)).convert("RGB")
            IM = self.transf(im)
            images.append(IM)
        tensor = torch.stack(images,dim=0)
        classification = self.label_index[label]
        return tensor, classification
        

    def _get_transforms(self, datatype):
        '''
        Get data augmentation processes
        '''

        input_size = 224

        if datatype == 'train':
            return transforms.Compose([\
                    transforms.RandomHorizontalFlip(),\
                    transforms.RandomVerticalFlip(),\
                    transforms.ColorJitter(brightness=0.5),\
                    transforms.RandomResizedCrop(input_size),\
                    transforms.ToTensor(),\
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            return transforms.Compose([\
                    transforms.CenterCrop(input_size),\
                    transforms.ToTensor(),\
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                
    def get_file_names_labels(self,idx):
        setnumbers = self.df['Set'].unique()
        df = self.df[self.df['Set'] == setnumbers[idx]]
        names = []
        label = df.iloc[0]['Label']
        directory = df.iloc[0]['directory']
        for mag in self.magnification:
            for m in self.mode:
                cond1 = df['Mode'] == m
                cond2 = df['Mag']== mag
                df1 = df[cond1*cond2]
                names.append(df1.iloc[0]['file'])
        
        return names, label, directory
    
    def create_dataloader(df, ROUTES, magnification, mode, datatype, batch_size, num_workers):
        dataset = MultiImage(df, ROUTES,magnification, mode, datatype)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)



