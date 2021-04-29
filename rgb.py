import numpy as np
import math
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,datasets
import os
import datetime
from PIL import Image

class KittiDataset(Dataset):
    def __init__(self,sequence,transform=None, left=False):       
        # only store images address in dataloader 
        side = 2 if left == True else 3
        folder = f'kitti/sequences/{sequence}/image_{side}'
        imgs = os.listdir(folder)
        self.images = [os.path.join(folder,img) for img in imgs if img != "desktop.ini"]
        self.images.sort()
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        if self.transform:
            image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.images)

if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    training_seq = ['00','02','08','09']

    nimages = 0
    mean = 0.0
    var = 0.0

    for seq in training_seq:
        print('{} Sequence {} initialized.'.format(datetime.datetime.now(), seq))
        kitti_dataset = KittiDataset(seq,transform=transforms.ToTensor(),left=True)
        data_loader = DataLoader(kitti_dataset, batch_size=32, num_workers=8, pin_memory=True)
        for i, batch_target in enumerate(data_loader):
            print('Batches processed: {} / {}'.format(i+1, len(data_loader)), end='\r')
            # subtracting 0.5
            batch_target -= 0.5
            # moving to the selected device
            batch_target = batch_target.to(device=device)
            # rearrange batch to be the shape of [B, C, W * H]
            batch_target = batch_target.view(batch_target.size(0), batch_target.size(1), -1)
            # update total number of images
            nimages += batch_target.size(0)
            # compute mean and std
            mean += batch_target.mean(2).sum(0) 
            var += batch_target.var(2).sum(0)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)

    if device == 'cuda:0':
        mean = mean.to(device='cpu')
        std = std.to(device='cpu')
        
    print('mean[0]={}'.format(mean[0], '.60g'))
    print('mean[1]={}'.format(mean[1], '.60g'))
    print('mean[2]={}'.format(mean[2], '.60g'))

    print('std[0]={}'.format(std[0], '.60g'))
    print('std[1]={}'.format(std[1], '.60g'))
    print('std[2]={}'.format(std[2], '.60g'))
