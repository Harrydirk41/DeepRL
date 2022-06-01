import torch
from torch.autograd import Variable
import numpy as np


def var2ten(x, cuda = True, type = "float"):
    if cuda:
        floattensor = torch.cuda.FloatTensor
        longtensor = torch.cuda.LongTensor
        bytetensor = torch.cuda.ByteTensor
    else:
        floattensor = torch.FloatTensor
        longtensor = torch.LongTensor
        bytetensor = torch.ByteTensor

    if type == "float":
        x = Variable(floattensor(np.array(x,dtype=np.float64)).tolist())
    if type == "long":
        x = Variable(longtensor(np.array(x, dtype=np.long)).tolist())
    if type == "byte":
        x = Variable(bytetensor(np.array(x, dtype=np.byte)).tolist())

def one_hot(action,dim):
    if isinstance(action,np.int) or isinstance(action,np.int64):
        one_hot = np.zeros(dim)
        one_hot[action] = 1
    else:
        one_hot = np.zeros((len(action),dim))
        one_hot[np.arange(len(action)),action] = 1
    return one_hot

def entropy(x):
    return -torch.sum(x * torch.log(x), 1)