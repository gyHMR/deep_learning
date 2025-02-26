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
dataset=datasets.ImageFolder(data_dir,transform=my_transform['val'])
train_data,val_data=random_split(dataset,[int(0.8*len(dataset)),int(0.2*len(dataset))])
train_dataloader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
val_dataloader=DataLoader(dataset=val_data,batch_size=batch_size,shuffle=True)

# 创建网络
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net=Googlenet(num_classes=num_classes).to(device)
net.load_state_dict(torch.load("./save_model/best_model.pth"))


# 开始评估模型
net.eval()
sum_loss,sum_acc,n=0,0,0
with torch.no_grad():
    for batch,(x,y) in enumerate(val_dataloader):
        x,y=x.to(device),y.to(device)
        output=net(x)
        _,pred=torch.max(output,dim=1)
        acc = torch.sum(pred == y).item() / y.shape[0]
        n+=1
        sum_acc+=acc
    print(f"准确率为：{sum_acc/n}")
