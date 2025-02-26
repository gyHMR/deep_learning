import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader


def load_fashion_mnist(batch_size):
    # 下载数据
    trans=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])  # 将图片转化为张量
    train_set=torchvision.datasets.FashionMNIST(root="./data",train=True,transform=trans,download=False)
    test_set=torchvision.datasets.FashionMNIST(root="./data",train=False,transform=trans,download=False)

    train_data=DataLoader(train_set,batch_size=batch_size,shuffle=True)
    test_data=DataLoader(test_set,batch_size=batch_size,shuffle=True)
    return train_data,test_data

# 设置超参数
batch_size=256
lr=0.01
num_epochs=10

train_iter,test_iter=load_fashion_mnist(batch_size)

num_inputs,num_outputs,num_hiddens=28*28,10,256