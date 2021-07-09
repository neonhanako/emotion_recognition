'''Train Fer2013 with PyTorch.'''
###
# 10 crop for data enhancement
from __future__ import print_function
import pandas as pd
# import transforms as transforms
import numpy as np
import os
from PIL import Image
# import utils
from torchvision import datasets,transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as Data
from torchvision import models
# from networks import *
import time
import torch
import torch.nn as nn
# from bayes_network import xception_layerfinall as target_network
# from bayes_network import resnet_layer4_ty1 as target_network
from bayes_network import resnet_layer4 as target_network
import argparse
import torch.optim as optim
from torchvision import datasets,transforms
import torch.nn.functional as F
import torchvision
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Data
print('==> Preparing data..')
Affect_transform_test = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((112,112)),transforms.ToTensor()
    ])

# set the testing data of aff-wild2
TEST_DATASET_PATH='./data/aff-wild2/test/'
filelist = os.listdir(TEST_DATASET_PATH)
filelist.sort()
# set the output path
result_path='./aff-wild2-result/'

parser = argparse.ArgumentParser(add_help=False)

# UCB HYPER-PARAMETERS
parser.add_argument('--samples', default='3', type=int, help='Number of Monte Carlo samples')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
args.sig1=0.0
args.sig2=6.0
args.pi=0.25
args.rho=-3
# Model
print(device)
net = target_network.Net(args).to(args.device)

print('==> Resuming from trained..')
savepath = './model'
checkpoint = torch.load(os.path.join(savepath,'aff-wild2_conv5_resnet18.t7'))
net.load_state_dict(checkpoint['net'], strict=True)

net.eval()
print('Test Start\n')

Affect_transform_val = transforms.Compose([transforms.Resize((112,112)), transforms.ToTensor()])

for i in range(len(filelist)):
    result_txt = filelist[i] + '.txt'
    with open(result_path+result_txt, 'w') as file:
        file.write('Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise' + '\n')

    flist = os.listdir(TEST_DATASET_PATH+filelist[i])
    flist.sort()
    for j in range(len(flist)):
        filename=flist[j]
        ext_str=os.path.splitext(filename)[1]
        if ext_str not in ['.jpg','.png','.jpeg','.tif','.tiff','.bmp','.gif']:
            continue
        predictions = []
        img_path=TEST_DATASET_PATH+filelist[i]+'/'+flist[j]
        # input = cv2.imread(img_path)
        with open(img_path, 'rb') as f:
            input = Image.open(f)
            input = input.convert('RGB')
        input = Affect_transform_val(input)
        input = input.unsqueeze(0)
        input = input.to(device)
        with torch.no_grad():
            for k in range(args.samples):
                outputs = net(input)
                predictions.append(outputs)

        outputs = torch.stack(predictions, dim=0).to(device)
        output = outputs.mean(0)

        _, predicted = torch.max(output.data, 1)
        predicted_ind = int(predicted.item())
        with open(result_path+result_txt, 'a+') as file:
            file.write(str(predicted_ind) + '\n')

        print('dir: %d/%d, file: %d/%d' % (i, len(filelist), j, len(flist)))

        temp=1



