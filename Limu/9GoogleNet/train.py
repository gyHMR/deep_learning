import os
import torch
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import random_split,DataLoader
from model import Googlenet


# 设置超参数
batch_size=64
lr=0.001
epochs=10
num_classes=10

#数据处理
my_transform={
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
}

# 加载数据  # 先加载整个数据集再进行随机划分   8:2进行划分
data_dir='./data/flower_photos'
dataset=datasets.ImageFolder(data_dir,transform=my_transform['train'])
train_data,val_data=random_split(dataset,[int(0.8*len(dataset)),int(0.2*len(dataset))])
train_dataloader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
val_dataloader=DataLoader(dataset=val_data,batch_size=batch_size,shuffle=True)

# 创建网络
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net=Googlenet(num_classes=num_classes).to(device)

# 损失函数和优化器
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=lr)


# 开始训练
def train(net,train_dataloader,loss,optimizer):
    sum_loss,sum_acc,n=0,0,0
    net.train()
    for batch,(x,y) in enumerate(train_dataloader):
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad()
        y_hat,au1,au2=net(x)
        # 计算准确率
        _, pred = torch.max(y_hat, dim=1)
        acc = torch.sum(pred == y).item() / y.shape[0]
        # 损失计算
        l1=loss(y_hat,y)
        l2=loss(au1,y)
        l3=loss(au2,y)
        #整合整体的损失
        l=l1+l2*0.3+l3*0.3
        l.backward()
        optimizer.step()

        sum_loss += l.item()
        sum_acc += acc
        n += 1

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


