import numpy as np
import torch
from torch.utils import data
from torch import nn
import torchvision
from torch.utils.data import DataLoader

# 累加器
class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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
lr=0.1
epoches=3

# 装载数据
train_data,test_data=load_fashion_mnist(batch_size)
# 回归模型
net=nn.Sequential(nn.Flatten(),nn.Linear(784,10))
def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
net.apply(init_weights)
# 损失函数
loss=nn.CrossEntropyLoss()
# 优化器
updater=torch.optim.SGD(net.parameters(),lr=lr)

# 准确率计算
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# 训练函数
def train_epoch(net, train_data, loss, updater):
    net.train()
    metric = Accumulator(3)  # 用于计算训练过程中的指标
    for x,y in train_data:
        y_hat=net(x)
        l=loss(y_hat,y)
        updater.zero_grad()
        l.backward()
        updater.step()
    with torch.no_grad():
        metric.add(l * x.shape[0], accuracy(y_hat, y), x.shape[0])
    # 返回训练损失和准确率
    return metric[0]/metric[2], metric[1]/metric[2]

# 测试函数
def evaluate_net(net,test_iter):
    net.eval()
    metric=Accumulator(2)
    with torch.no_grad():
        for x,y in test_iter:
            y_hat=net(x)
            metric.add(accuracy(y_hat,y),y.numel())
    return metric[0]/metric[1]


# 训练和评估模型
def train(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型并在测试集上评估"""
    train_loss, train_accuracy = [], []
    test_accuracy = []

    for epoch in range(num_epochs):
        # 训练阶段
        train_l, train_acc = train_epoch(net, train_iter, loss, updater)
        # 测试阶段
        test_acc = evaluate_net(net, test_iter)
        # 记录数据
        train_loss.append(train_l)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)
        # 打印结果
        print(f'Epoch {epoch + 1}, '
              f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}')

# 调用训练函数
train(net, train_data, test_data, loss, epoches, updater)