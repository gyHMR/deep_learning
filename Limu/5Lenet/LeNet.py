import torch
from torch import nn


class leNet(nn.Module):
    def __init__(self):
        # super：引入父类的初始化方法给子类进行初始化
        super(leNet,self).__init__()
        # 开始写层数
        self.model=nn.Sequential(
            # c1卷积层  28*28——》28*28 （28-5+2*2）/1+1
            nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2),
            nn.Sigmoid(),  #非线性
            # 平均池化层s2 28*28——》14*14 （28-2）/2+1
            nn.AvgPool2d(kernel_size=2,stride=2),
            #c3卷积层 14*14——》10*10 （14-5）/1+1
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5),
            nn.Sigmoid(),  # 非线性
            #s4池化层 10*10——》5*5 （10-2）/2+1
            nn.AvgPool2d(kernel_size=2,stride=2),
            #c5卷积层 5*5——》1*1 （5-5）/1+1
            nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5),
            #得到的是一个batchsize*120*1*1的四维张量，拉伸成1维的batchsize*120
            nn.Flatten(),
            #全连接层
            nn.Linear(120,84),
            #输出层
            nn.Linear(84,10)
        )

    def forward(self,x):
        x=self.model(x)
        return x

# def test():
#     x=torch.rand(1,1,28,28)
#     net=LeNet()
#     print(net(x).shape)
# test()



