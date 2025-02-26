import os

import torch
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from model import Vgg


# 设置超参数
batch_size=32
lr=0.001
epochs=10



# 数据格式处理
my_transform=transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]
)
# 下载数据
train_data=datasets.CIFAR10(root='./data',train=True,transform=my_transform,download=True)
test_data=datasets.CIFAR10(root='./data',train=False,transform=my_transform,download=True)
# 装载数据
# test_dataloader=DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True)
train_dataloader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)

# 设置gpu上
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# 创建网络
net=Vgg().to(device)

# 损失函数和优化器
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=lr)

# 开始训练
def train(net,train_dataloader,loss,optioner):
    sum_loss,n,sum_acc=0,0,0
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

        sum_loss += l.item()
        sum_acc+=acc
        n+=1

    train_loss = sum_loss / n;
    train_acc = sum_acc / n
    print(f"训练损失是：{train_loss}")
    print(f"训练准确率为：{train_acc}")
    return train_acc


min_acc=0

for epoch in range(epochs):
    print(f"第{epoch+1}轮训练:")
    cur_acc=train(net,train_dataloader,loss,optimizer)
    # 保存最好的模型
    if(cur_acc>min_acc):
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir(folder)
        min_acc = cur_acc
        print('save best model')
        torch.save(net.state_dict(), 'save_model/best_model.pth')

print("训练结束")



