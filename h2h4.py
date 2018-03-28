from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import time 
import math
import pandas as pd
import sys
import os

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=8192, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--dim', type=int, default=50, metavar='N')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
args = parser.parse_args()
version = 106

BATCH_SIZE = args.batch_size
d = args.dim
LR = args.lr
npasses = 1



B = Variable(torch.randn(d,d).cuda(), requires_grad = True)
#B = Variable(torch.eye(d).cuda(), requires_grad = True)
B.data = B.data.div(B.data.norm(p=2,dim=1, keepdim=True))
TRUE_B = Variable(torch.eye(d).cuda())
optimizer = optim.SGD([B], lr=LR)

def h2(x):
    return (x.pow(2)-1.0)/math.sqrt(2)

def h4(x):
    return (x.pow(4) - 6.0*x.pow(2) + 3.0)/math.sqrt(24)

def h2h4(x):
    sigma2 = 1.0/(2*math.sqrt(math.pi))
    sigma4 = -1.0/(4*math.sqrt(math.pi*3))
    return sigma2*h2(x)+sigma4*h4(x)

def loss(y,X,B):
    yhat = Variable(torch.ones(1,d).cuda()).mm(h2h4(torch.mm(B,X)))
    l = (y-yhat).pow(2).mean()/d
    return l

def true_model(B,X):
    return Variable(torch.ones(1,d).cuda()).mm(F.relu(torch.mm(B,X)))
    
def train_label(batch_size, B, X):
    #X = Variable(torch.randn(d,batch_size)).cuda()
    y = true_model(B,X)
    return y

def train_batch(X, batch_size=BATCH_SIZE):
    start = time.time()
    y= train_label(batch_size, TRUE_B, X)
    l = loss(y,X,B)
    re = l.data.cpu().numpy()
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    B.data = B.data.div(B.data.norm(p=2,dim=1, keepdim=True))
    return re, time.time()-start

def generate_data(size):
    print('generating data ...')
    return np.random.normal(0, 1, [d, size])

def train_epoch(full_data):
    start_time = time.time()
    n = full_data.size()[1]
    loss_list = []
    for i in range(math.floor(n/BATCH_SIZE)):
        start = i*BATCH_SIZE
        end = (i+1)*BATCH_SIZE
        X = Variable(full_data[:,start:end])
        batch_loss, duration = train_batch(X)
        loss_list += [batch_loss[0]]
    return loss_list, time.time() -start_time
        
def train_fixed_data(ndata, nepoch):
    print('generating data...')
    start_time = time.time()
    full_data = np.random.normal(0,1,[d,ndata])
    full_data = torch.FloatTensor(full_data).cuda()
    print('time to generate data', time.time()-start_time)
    loss_list = []
    for i in range(nepoch):
        l, duration= train_epoch(full_data)
        loss_list += l
        Beval = B.data.cpu().numpy()
        print('h2h4: averag_loss = %.5f, std = %.3f, row_norm = %3.f, error = %.3f, error2 = %.3f, %.3f sec/epoch, %.3f sec/1M, lr = %.3E, b = %d, d=%d'%(np.mean(l),
                                                np.std(l)/np.sqrt(len(l)),
                                                np.mean(np.linalg.norm(Beval,axis=1)),
                                                1- np.min(np.amax(np.abs(Beval),axis=1)), 
                                                1- np.min(np.amax(np.abs(Beval),axis=0)),
                                                duration, 
                                                duration*1000000/ndata,
                                                LR,
                                                BATCH_SIZE,
                                                d))
        if i == 0:
            test_error = np.mean(l)
            error1 = 1 - np.min(np.amax(np.abs(Beval),axis=1))
            error2 = 1 - np.min(np.amax(np.abs(Beval),axis=0))
    return test_error, error1, error2

loss_list = []
error_list = []
error2_list = []
for _ in range(30000):
    test_error, error1, error2 = train_fixed_data(int(BATCH_SIZE*500), npasses)
    loss_list += [test_error]
    error_list += [error1]
    error2_list += [error2]
    print(error1)
    print(error_list)
    print(error2_list)
    print(loss_list)
    data = {}
    data['loss'] = loss_list
    data['error'] = error_list
    data['error2'] = error2_list
    df = pd.DataFrame(data=data)
    directory = 'results'
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_csv(directory + '/' + 'h2h4_d%d_batch%d_LR%f_pass%d_run%d.csv'%(d,BATCH_SIZE,LR,npasses,version))
    