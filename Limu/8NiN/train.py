import os

import torch
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from model import NiN

#超参数设置
batch_size=64   #太大了占内存，太小了收敛速度慢
lr=0.01    #太大不容易收敛，太小了降低训练速度
epochs=20

# 图像尺寸处理
my_transform=transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ]
)

# 下载数据
train_data=datasets.FashionMNIST(root="./data",train=True,transform=my_transform,download=True)
test_data=datasets.FashionMNIST(root="./data",train=False,transform=my_transform,download=True)
# 装载数据
train_dataloader=DataLoader(train_data,batch_size=batch_size,shuffle=True)

# 创建网络
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=NiN()
net.to(device)
# 损失函数和优化器
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=lr)


# 开始训练
def train(net,train_dataloader,loss,optioner):
    sum_loss,sum_acc,n=0,0,0
    net.train()
    for batch,(x,y) in enumerate(train_dataloader):
        x,y=x.to(device),y.to(device)
        optioner.zero_grad()
        y_hat=net(x)
        l=loss(y_hat,y)
        _,pred=torch.max(y_hat,dim=1)
        acc=torch.sum(pred==y).item()/y.shape[0]
        l.backward()
        optioner.step()

        sum_loss+=l.item()
        sum_acc+=acc
        n+=1
    print(f"第{batch}")
    print(f"训练损失是：{sum_loss/n:.4f}")
    print(f"训练精度是：{sum_acc / n:.4f}")
    return sum_acc/n

min_acc=-1;
for epoch in range(epochs):
    print(f"第{epoch + 1}轮训练:")
    cur_acc = train(net, train_dataloader, loss, optimizer)
    # 保存最好的模型
    if (cur_acc > min_acc):
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir(folder)
        min_acc = cur_acc
        print('save best model')
        torch.save(net.state_dict(), 'save_model/best_model.pth')

print("训练结束")