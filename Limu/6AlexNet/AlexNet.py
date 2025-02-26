import torch
from torch import nn


class alexnet(nn.Module):
    def __init__(self):
        super(alexnet, self).__init__()
        self.modle=nn.Sequential(
            # 实际情况下用的是224*224*1
            # c1卷积层227*227*3->55*55*96  (227-11)/4+1=55
            nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),  #55*55*96->27*27*96 (55-3)/2+1=27
            #c2卷积层27*27*96->27*27*256  (27-5+2*2)/1+1=27
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2), #27*27*256->13*13*256 (27-3)/2+1=13
            #c3卷积层13*13*256->13*13*384  (13-3+2*1)/1+1=13
            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1),
            nn.ReLU(),
            #c4卷积层13*13*384->13*13*384  (13-3+2*1)/1+1=13
            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1),
            nn.ReLU(),
            #c5卷积层13*13*384->13*13*256  (13-3+2*1)/1+1=13
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2), #13*13*256->6*6*256 (13-3)/2+1=6
            #FC6全连接层
            nn.Flatten(),
            nn.Linear(6400,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            #FC7全连接层
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            #FC8全连接层
            nn.Linear(4096,10)
        )

    def forward(self,x):
        x=self.modle(x)
        return x


# model=alexnet()
# x=torch.rand(1,1,224,224)
# for layer in model.modle:
#     x=layer(x)
#     print(layer.__class__.__name__,x.shape)