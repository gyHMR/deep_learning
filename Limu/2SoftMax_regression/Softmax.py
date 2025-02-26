import torchvision
import torch
from torch.utils.data import DataLoader



# 装载数据
def load_fashion_mnist(batch_size):
    # 下载数据
    trans=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])  # 将图片转化为张量
    minist_data=torchvision.datasets.FashionMNIST(root="./data",train=True,transform=trans,download=False)
    minist_data=torchvision.datasets.FashionMNIST(root="./data",train=False,transform=trans,download=False)

    train_data=DataLoader(minist_data,batch_size=batch_size,shuffle=True)
    test_data=DataLoader(minist_data,batch_size=batch_size,shuffle=True)
    return train_data,test_data

# 定义模型参数
num_inputs=28*28 # 输入维度（softmax要求一维张量）
num_outputs=10 #数据集有10个类别
w=torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b=torch.zeros(num_outputs,requires_grad=True)


# 定义模型
def softmax(x):
    x_exp=torch.exp(x)
    partition=x_exp.sum(1,keepdim=True)  #将所有列相加
    return x_exp/partition
def net(x):
    return softmax(torch.matmul(x.reshape(-1,num_inputs),w)+b)

# 交叉熵损失函数
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])  #取出y_hat中y对应的列，即预测正确的概率


# 优化算法

batch_size=64
train_data,test_data=load_fashion_mnist(batch_size=batch_size)


