import os

import  torch
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import  DataLoader

from AlexNet import alexnet

#设置超参数
lr=0.01
epochs=10
batch_size=128


# 图像处理
my_transform=transforms.Compose([
    transforms.Resize((224,224)),  #裁剪到224*224
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# 下载数据集
train_data=datasets.FashionMNIST(root="./data",train=True,transform=my_transform,download=True)
test_data=datasets.FashionMNIST(root="./data",train=False,transform=my_transform,download=True)

# 加载数据集
train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_data=DataLoader(test_data,batch_size=batch_size,shuffle=True)

# 创建网络
device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
net=alexnet().to(device)

# 设置损失函数和优化器
loss=nn.CrossEntropyLoss()
optioner=torch.optim.SGD(net.parameters(),lr=lr)

# 开始训练
def train(net,train_loader,loss,optioner):
    sum_loss,sum_acc,n=0,0,0
    net.train()
    for batch,(x,y) in enumerate(train_loader):
        x,y=x.to(device),y.to(device)
        optioner.zero_grad()
        y_hat=net(x)
        if n==0:
            print("第一次训练完毕")
        l=loss(y_hat,y)
        _,pred=torch.max(y_hat,dim=1)
        acc=torch.sum(pred==y).item()/y.shape[0]

        # 反向传播
        l.backward()
        optioner.step()

        sum_loss+=l.item()
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
    cur_acc=train(net,train_loader,loss,optioner)
    # 保存最好的模型
    if(cur_acc>min_acc):
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir(folder)
        min_acc = cur_acc
        print('save best model')
        torch.save(net.state_dict(), 'save_model/best_model.pth')

print("训练结束")
