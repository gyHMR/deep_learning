import os

import torch
import torchvision
from torch import nn
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from LeNet import leNet

# 设置超参数
lr=0.1
epochs=10
batch_size=16

# 获取数据
# 首先将数据转换为tensor格式   将读取到的图片转换为c,h,w的tensor形式
my_transform=transforms.Compose([transforms.ToTensor()])
train_data=datasets.MNIST(root="./data",train=True,transform=my_transform,download=True)
test_data=datasets.MNIST(root='./data',train=False,transform=my_transform,download=True)

# 加载数据集
train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_data,batch_size=batch_size,shuffle=True)

# 创建网络并设置用gpu训练
device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
net=leNet().to(device)




# 损失函数
loss=nn.CrossEntropyLoss()
# 优化器
optioner=torch.optim.SGD(net.parameters(),lr=lr)


def train(net,dataloader,loss,optioner):
    sum_loss,sum_acc,n=0,0,0
    net.train()
    for batch,(x,y) in enumerate(dataloader):
        x,y=x.to(device),y.to(device)
        # 梯度清零
        optioner.zero_grad()
        # 计算训练值
        output=net(x)
        # 计算损失
        l=loss(output,y)
        # 得到训练数据中每行的最大值和对应的索引
        _,pred=torch.max(output,dim=1)
        acc=torch.sum(pred==y).item()/y.shape[0]

        # 反向传播
        l.backward()
        #更新参数
        optioner.step()

        # 计算误差，准确率等
        sum_loss+=l.item()
        sum_acc+=acc
        n+=1
    train_loss=sum_loss/n;
    train_acc=sum_acc/n
    print(f"训练损失是：{train_loss}")
    print(f"训练准确率为：{train_acc}")
    return train_acc


# 开始训练
min_acc=0
for epoch in range(epochs):
    print(f"第{epoch+1}轮训练:")
    cur_acc=train(net,train_loader,loss,optioner)
    # 保存最好的模型
    if(cur_acc>min_acc):
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir(folder)
        min_acc = cur_acc
        print('save best model')
        # torch.save(state, dir)保存模型等相关参数，dir表示保存文件的路径+保存文件名
        # model.state_dict()：返回的是一个OrderedDict，存储了网络结构的名字和对应的参数
        torch.save(net.state_dict(), 'save_model/best_model.pth')

print("训练结束")
