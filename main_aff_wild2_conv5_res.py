'''Train Fer2013 with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function
import pandas as pd
# import transforms as transforms
import numpy as np
import os
# import utils
from torchvision import datasets,transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as Data
from torchvision import models
from torch.autograd import Variable
# from networks import *
#from myOperation import  get_kl,get_kl_loss
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
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

resume=0
pre_train=0
# Hyper parameters
dataset='FER2013'
bs=128
bs_=64

start_epoch = 1  # start from epoch 0 or last checkpoint epoch
total_epoch =400  # 246#300

Train_acc=0
totalacc=0
best_PublicTest_acc = 0 # best PublicTest accuracy
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0


# 用小数据集就用下面代码
# training_sheet = pd.read_csv('training.csv')
# training_sheet_split = pd.DataFrame(training_sheet.subDirectory_filePath.str.split("/").tolist(),
#                                     columns=['folder', 'subpath'])
# folders = list(map(int, training_sheet_split.folder))
# folder_list = [1, 10, 100, 102, 103] + list(range(1000, 1030))
# inFolder = np.isin(folders, folder_list)


path= 'new_model'

# Data
print('==> Preparing data..')
#       transforms.ToPILImage(),
Affect_transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((112,112)),transforms.ToTensor()
    ])

Affect_transform_val = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((112,112)),transforms.ToTensor()
    ])

TRAIN_DATASET_PATH='/home/ubuntu/gongwei/emotion_recognition/data/aff-wild2/training/'
VAL_DATASET_PATH='/home/ubuntu/gongwei/emotion_recognition/data/aff-wild2/evaluation/'

train_dataset = ImageFolder(TRAIN_DATASET_PATH, transform=Affect_transform_train)
train_num=train_dataset.__len__()
print('Train set size:', train_num)
trainloader = Data.DataLoader(dataset=train_dataset, batch_size=bs, num_workers=4, shuffle=True)
PublicTestset = ImageFolder(VAL_DATASET_PATH, transform=Affect_transform_val)
print('Validation set size:', PublicTestset.__len__())
valloader = Data.DataLoader(dataset=PublicTestset, batch_size=bs, num_workers=4, shuffle=False)


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--nlayers', default=1, type=int, help='')
parser.add_argument('--nhid', default=1200, type=int, help='')
parser.add_argument('--parameter', default='', type=str, help='')
parser.add_argument('--lr', default=0.001, type=str, help='')

# UCB HYPER-PARAMETERS
parser.add_argument('--samples', default='3', type=int, help='Number of Monte Carlo samples')
parser.add_argument('--rho', default='-3', type=float, help='Initial rho')
parser.add_argument('--sig1', default='0.0', type=float,
                    help='STD foor the 1st prior pdf in scaled mixture Gaussian')
parser.add_argument('--sig2', default='6.0', type=float,
                    help='STD foor the 2nd prior pdf in scaled mixture Gaussian')
parser.add_argument('--pi', default='0.25', type=float, help='weighting factor for prior')
parser.add_argument('--arch', default='mlp', type=str, help='Bayesian Neural Network architecture')

parser.add_argument('--resume', default='yes', type=str, help='resume?')
parser.add_argument('--sti', default=1, type=int, help='starting task?')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
# Model
print(device)
net = target_network.Net(args).to(args.device)

# net = target_network.DenseNet(args,growthRate=12, depth=100, reduction=0.5,
#                         bottleneck=True, nClasses=10).to(device)

# net = ResNet(BasicBlock, [2,2,2,2])

# net = ResNet_BNN_layers2(BasicBlock,[2,2,2,2],device)
# net = VGG19_BNN_conv1('VGG19',device)
# var_model = pyvarinf.Variationalize(net)

# net = torch.nn.DataParallel(net)

# Load checkpoint. 加载原有的模型继续训练
if resume:
    print('==> Resuming from checkpoint..')
    savePath=path
    checkpoint = torch.load(os.path.join(savePath,'aff-wild2_conv5_resnet18.t7'),map_location=device)
    net.load_state_dict(checkpoint['net'],strict=True)
    best_PublicTest_acc = checkpoint['acc'].cpu()
    # best_PrivateTest_acc = checkpoint['best_PrivateTest_acc']
    # best_PublicTest_acc_epoch = checkpoint['best_PublicTest_acc_epoch']
    # best_PrivateTest_acc_epoch = checkpoint['best_PrivateTest_acc_epoch']
    start_epoch = checkpoint['epoch'] + 1
    print(start_epoch)
    print(best_PublicTest_acc)
# if resume:
#     print('==> Resuming from resume..')
#     savepath='FER_models\VGG19_BNN\\2020-04-06_13-02-02/train_model.t7'
#     net.load_state_dict(torch.load(savepath),strict=False)
if pre_train:
    print('==> Resuming from pre_trained..')
    savepath = '/home/ubuntu/gongwei/emotion_recognition/CIF_Xception_final_test.t7'
    checkpoint = torch.load(savepath)
    net.load_state_dict(checkpoint['net'], strict=False)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer=torch.optim.RMSprop(net.parameters(),lr=lr,alpha=0.9)
# optimizer = optim.SGD(net.parameters(), lr= lr, momentum=0.9 , weight_decay=5e-4)
# optimizer = optim.Adam(net.parameters(),lr=lr)


def find_uncertain_names(model):
    modules_names = []
    for name, p in model.named_parameters():
        if name.endswith('_mu') or name.endswith('_rho') :
            n = name.split('.')[:-1]
            modules_names.append('.'.join(n))
    modules_names = set(modules_names)
    return modules_names

def find_certain_names(model):
    modules_names = []
    for name, p in model.named_parameters():
        if not name.endswith('_mu') and not name.endswith('_rho'):
            n = name.split('.')[:-1]
            modules_names.append('.'.join(n))
    modules_names = set(modules_names)
    return modules_names

def logs():
    lp, lvp = 0.0, 0.0
    for name in find_uncertain_names(net):
        n = name.split('.')
        if len(n) == 1:
            m = net._modules[n[0]]
        elif len(n) == 2:
            m = net._modules[n[0]]._modules[n[1]]
        elif len(n) == 3:
            m = net._modules[n[0]]._modules[n[1]]._modules[n[2]]
        elif len(n) == 4:
            m = net._modules[n[0]]._modules[n[1]]._modules[n[2]]._modules[n[3]]

        lp += m.log_prior
        lvp += m.log_variational_posterior
    # lp += net.classifier.log_prior
    # lvp += net.classifier.log_variational_posterior
    return lp, lvp

def uncertain_par(models,lr,adaptive_lr=False):
    params_dict = []

    for name in find_uncertain_names(net):

        n = name.split('.')
        if len(n) == 1:
            m = models._modules[n[0]]
        elif len(n) == 2:
            m = models._modules[n[0]]._modules[n[1]]
        elif len(n) == 3:
            m = models._modules[n[0]]._modules[n[1]]._modules[n[2]]
        elif len(n) == 4:
            m = models._modules[n[0]]._modules[n[1]]._modules[n[2]]._modules[n[3]]
        else:
            print (name)

        if adaptive_lr is True:
            params_dict.append({'params': m.weight_rho, 'lr': lr})
            params_dict.append({'params': m.bias_rho, 'lr': lr})
            params_dict.append({'params': m.weight_mu, 'lr': lr})
            params_dict.append({'params': m.bias_mu, 'lr': lr})

        else:
            params_dict.append({'params': m.weight_rho, 'lr':lr})
            # params_dict.append({'params': m.bias_rho, 'lr':lr})
            params_dict.append({'params': m.weight_mu, 'lr': lr})
            # params_dict.append({'params': m.bias_mu, 'lr': lr})

    return params_dict


def certain_par(models,lr,adaptive_lr=False):
    params_dict = []

    for name in find_certain_names(net):

        n = name.split('.')
        if len(n) == 1:
            m = models._modules[n[0]]
        elif len(n) == 2:
            m = models._modules[n[0]]._modules[n[1]]
        elif len(n) == 3:
            m = models._modules[n[0]]._modules[n[1]]._modules[n[2]]
        elif len(n) == 4:
            m = models._modules[n[0]]._modules[n[1]]._modules[n[2]]._modules[n[3]]
        else:
            print (name)

        if adaptive_lr is True:

            params_dict.append({'params': m.weight, 'lr': lr})
            # params_dict.append({'params': m.bias, 'lr': lr})

        else:
            params_dict.append({'params': m.weight, 'lr': lr})
            # params_dict.append({'params': m.bias, 'lr': lr})

    return params_dict
# un_parameters = find_uncertain_names(net)
# parameters = find_uncertain_names(net)

# optimizer = optim.Adam(net.parameters(), lr=args.lr)

un_parameters=uncertain_par(net,args.lr)
parameters=certain_par(net,args.lr)

un_optimizer = optim.Adam(un_parameters, lr=args.lr, weight_decay=1e-4)
optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# Training
def train(epoch):
    global Train_acc

    net.train()

    train_loss = 0
    train_loss1 = 0
    train_loss2 = 0
    train_loss3 = 0
    correct = 0
    total = 0

    print('learning_rate: %s' % str(args.lr))

    for batch_idx, (inputs, targets) in enumerate(trainloader): #

        un_optimizer.zero_grad()
        optimizer.zero_grad()

        inputs, targets = inputs.to(device), targets.to(device)
        # inputs, targets = Variable(inputs), Variable(targets)

        lps, lvps, predictions = [], [], []
        for i in range(args.samples):
            y_pred = net(inputs)
            predictions.append(y_pred)
            lp, lv = logs()
            lps.append(lp)
            lvps.append(lv)
        w1 = 1.e-3
        w2 = 1.e-3
        w3 = 5.e-2
        outputs = torch.stack(predictions, dim=0).to(device)
        output = outputs.mean(0)
        un_output=F.log_softmax(output, dim=1)

        _, pred = output.max(1, keepdim=True)
        # acc = accuracy(output.data, targets, topk=(1,))[0].item()

        log_var = w1 * torch.as_tensor(lvps, device=device).mean()
        log_p = w2 * torch.as_tensor(lps, device=device).mean()
        nll = w3 * torch.nn.functional.nll_loss(un_output, targets, reduction='sum').to(device=device)

        loss1 = criterion(output, targets)
        loss2 = (log_var - log_p)+nll

        loss2.backward(retain_graph=True)
        un_optimizer.step()

        loss1.backward(retain_graph=True)
        optimizer.step()

        # klweight = torch.sigmoid(net.klweight_)
        # loss = (1 - klweight) * loss1 + klweight * loss2

        loss1 +=loss1.item()
        loss2 += loss2.item()

        if torch.isnan(loss1) or torch.isnan(loss2):
            epoch=1000000
            print('Nan')

        _, predicted = torch.max(output.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        # print('RAFvgg_conv4 epoch: %d cross_loss %0.3f, kl_loss: %0.3f , Train_acc: %0.3f,[%d/%d]\n' % (epoch,loss1,loss2, correct.float()/total,batch_idx+1, len(trainloader)))
    # scheduler.step()
    Train_acc =  correct.float() / total
    print('aff-wild2_conv5_resnet18 epoch: %d cross_loss: %0.3f,   kl_loss: %0.3f , Av_Train_acc: %0.3f\n' % (epoch,loss1,loss2,Train_acc))

    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': Train_acc,
        'epoch': epoch,
        'best_PublicTest_acc':best_PublicTest_acc
    }

    if not os.path.isdir(path):
        os.mkdir(path)
    # torch.save(state, os.path.join(path, 'FER_RES_layer4_train.t7'))

    # if epoch % 10 == 0:
    file = open('aff-wild2_conv5_resnet18.txt', 'a+')
    file.writelines('aff-wild2_conv5_resnet18 epoch: %d   cross_loss %0.3f, kl_loss: %0.3f , Train_acc: %0.3f\n' % (epoch,loss1,loss2,Train_acc))
    file.close()
    # info = {'loss': train_loss/128, 'accuracy': Train_acc,
    #         'cross_loss': train_loss1/128, 'uncertainty': train_loss2/128, 'kl_loss': train_loss3/128}


def PublicTest(epoch):
    global PublicTest_acc
    global best_PublicTest_acc
    global best_PublicTest_acc_epoch
    global totalacc
    net.eval()
    print('Test Start\n')
    correct = 0
    total = 0
    total_epistemic=0.
    total_aleatoric=0.

    epistemic=0
    for batch_idx, (inputs, targets) in enumerate(valloader):
        lps, lvps, predictions = [], [], []
        ##################### ten_cropped
        # bs, ncrops, c, h, w = np.shape(inputs)
        # inputs = inputs.view(-1, c, h, w)
        # if use_cuda:
        #     inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            for i in range(args.samples):
                outputs = net(inputs)

                predictions.append(outputs)

        outputs = torch.stack(predictions, dim=0).to(device)
        output = outputs.mean(0)

        output_ep = F.softmax(output) + 0.00001
        epistemic = epistemic - (output_ep * output_ep.log()).cpu().sum()

        # outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        _, predicted = torch.max(output.data, 1)

        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # total_epistemic +=epistemic
        # total_aleatoric +=aleatoric

        # print('RAFres_conv4 epoch: %d , Test_acc: %0.3f,best_PublicTest_acc %0.3f，[%d/%d]' % (epoch, correct.float() / total,best_PublicTest_acc, batch_idx + 1, len(valloader)))

    PublicTest_acc = correct.float() / total
    av_total_epistemic = epistemic.float() / total
    # av_total_epistemic=total_epistemic/total
    # av_total_aleatoric = total_aleatoric / total
    print('aff-wild2_conv5_resnet18 Epoch: %d RAF_val_acc: %0.3f   best_PublicTest_acc： %0.3f  av_total_epistemic: %0.3f' %(epoch,PublicTest_acc,best_PublicTest_acc,av_total_epistemic))
    totalacc += PublicTest_acc

    file1 = open('aff-wild2_conv5_resnet18_accuracy.txt', 'a+')
    file1.write(str(epoch)+'  '+str(PublicTest_acc)+'\n')
    file1.close()
    file2 = open('aff-wild2_conv5_resnet18_epistemic.txt', 'a+')
    file2.write(str(epoch)+'  '+str(av_total_epistemic)+'\n')
    file2.close()

    if PublicTest_acc > best_PublicTest_acc:
        print('Saving..')
        print("best_PublicTest_acc: %0.3f" % PublicTest_acc)
        file = open('aff-wild2_conv5_resnet18.txt', 'a+')
        file.writelines('best_Val_acc: %0.3f\n' % (PublicTest_acc))
        file.writelines('av_total_epistemic: %0.3f\n' % (av_total_epistemic))
        file.close()
        state = {
            'net': net.state_dict(),
            'acc': PublicTest_acc,
            'epoch': epoch,
        }

        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'aff-wild2_conv5_resnet18.t7'))
        best_PublicTest_acc = PublicTest_acc
        best_PublicTest_acc_epoch = epoch


for epoch in range(start_epoch, total_epoch+1):

    train(epoch)
    PublicTest(epoch)
    # PrivateTest(epoch)

print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc)
print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)

