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


'''
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--dim', type=int, default=5, metavar='N')
parser.add_argument('--lr', type=float, default=0.03, metavar='LR',
                    help='learning rate (default: 0.01)')
args = parser.parse_args()
'''

version = 106

#FACTOR = 1
BATCH_SIZE = 1024
SAMPLE_SIZE = 262144
d = 5
LR = 1e-3
npasses = 1



B = Variable(torch.randn(d,d).cuda(), requires_grad = True)
B.data = B.data.div(B.data.norm(p=2,dim=1, keepdim=True))
TRUE_B = Variable(torch.eye(d).cuda())
optimizer = optim.SGD([B], lr=LR)


def loss(y,X,B, mu=5.0):
    m = d
    term1 = (0.5*m*m - 1.5*m)*Variable(torch.ones(1, BATCH_SIZE).cuda())
    temp = torch.mm(B,torch.t(B)).pow(2).sum(dim=0,keepdim=True).sum(dim=1,keepdim=True)
    term2 = torch.mm(temp, Variable(torch.ones(1,BATCH_SIZE).cuda()))
    BX = torch.mm(B,X)
    Bxnormsrt = BX.pow(2).sum(dim=0, keepdim=True)
    term3 = (3.0-m)*Bxnormsrt
    temp = B.t().mm(B).mm(B.t()).mm(B).mm(X)
    temp = temp.mul(X).sum(dim=0, keepdim=True)
    term4 = -2.0*temp
    term5 = 0.5*Bxnormsrt.pow(2)
    BXnorm4 = BX.pow(4).sum(dim=0, keepdim=True)
    term6 = -0.5*BXnorm4
    terms_sum = term1 + term2 + term3 + term4 + term5 + term6
    reg = 1.0*mu*y.mul(1/8.0*m*Variable(torch.ones(1,BATCH_SIZE).cuda()) - 1/4.0*Bxnormsrt + 1/24.0*BXnorm4).mean(dim=1)
    re = -1.0*y.mul(terms_sum).mean(dim=1) + reg
    return re/d

def loss_l2(y,X,B):
    hat_y = Variable(torch.ones(1,d).cuda()).mm(F.relu(torch.mm(B,X)))
    return torch.mean((y-hat_y).pow(2))

def true_model(B,X):
    return Variable(torch.ones(1,d).cuda()).mm(F.relu(torch.mm(B,X)))

def train_label(batch_size, B, X):
    #X = Variable(torch.randn(d,batch_size)).cuda()
    y = true_model(B,X)
    return y

def train_batch(X, batch_size=BATCH_SIZE):
    start = time.time()
    y= train_label(batch_size, TRUE_B, X)
    l = loss_l2(y,X,B)
    re = l.data.cpu().numpy()
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    B.data = B.data.div(B.data.norm(p=2,dim=1, keepdim=True))
    return re, time.time()-start

def generate_data(size):
    print('generating data ...')
    return np.random.standard_t(5, [d, size])

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
    #full_data = np.random.normal(0,1,[d,ndata])
    #full_data = torch.FloatTensor(full_data).cuda()
    #full_data = torch.normal(torch.zeros(d,ndata).cuda(), torch.ones(d,ndata).cuda())
    full_data = torch.FloatTensor(generate_data(ndata)).cuda()
    loss_list = []
    for i in range(nepoch):
        l, duration= train_epoch(full_data)
        loss_list += l
        Beval = B.data.cpu().numpy()
        print('averag_loss = %.3f, std = %.3f, error = %.3f, error2 = %.3f, %.3f sec/epoch, %.3f sec/1M, lr = %.3E, b = %d, d=%d'%(np.mean(l),
                                                np.std(l)/np.sqrt(len(l)),
                                                1- np.min(np.amax(np.abs(Beval),axis=1)),
                                                1- np.min(np.amax(np.abs(Beval),axis=0)),
                                                duration,
                                                duration*1000000/ndata,
                                                LR,
                                                BATCH_SIZE,
                                                d))
        if i == 0:
            test_error = np.mean(l)
            error1 = 1- np.min(np.amax(np.abs(Beval),axis=1))
            error2 = 1- np.min(np.amax(np.abs(Beval),axis=0))

    print('time for one epoch', time.time()-start_time)
    return test_error, error1, error2


loss_list = []
error_list = []
error2_list = []
for rnd in range(3000):
    print("Round: %d"%rnd)
    test_error, error1, error2 = train_fixed_data(SAMPLE_SIZE, npasses)

    loss_list += [test_error]
    error_list += [error1]
    error2_list += [error2]
    data = {}
    data['loss'] = loss_list
    data['error'] = error_list
    data['error2'] = error2_list
    df = pd.DataFrame(data=data)
    directory = 'results'
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_csv(directory + '/' + 'd%d_batch%d_LR%f_pass%d_run%d.csv'%(d,BATCH_SIZE,LR,npasses,version))
