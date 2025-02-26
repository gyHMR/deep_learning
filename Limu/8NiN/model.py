import torch
from torch import nn



# 一个卷积后跟两个全连接层
def nin_block(in_channels,out_channels,kernel_size,stride,padding):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
        nn.ReLU(),
        # 使用1*1卷积代替全连接层
        nn.Conv2d(out_channels,out_channels,kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
    )

class NiN(nn.Module):
    def __init__(self):
        super(NiN,self).__init__()
        self.model=nn.Sequential(
            # 224*224*1->54*54*96  (224-11)/4+1=53.25+1(一般来说，卷积操作向下取整)
            nin_block(1,96,kernel_size=11,stride=4,padding=0),
            # 54*54*96-》26*26*96 (54-3)/2+1=25.5+1=26
            nn.MaxPool2d(kernel_size=3,stride=2),
            # 26 * 26 * 96->26*26*256 (26-5+2*2)/1+1=26
            nin_block(96,256,kernel_size=5,stride=1,padding=2),
            # 26 * 26 * 256->12*12*256(26-3)/2+1=12
            nn.MaxPool2d(kernel_size=3,stride=2),
            # 12 * 12 * 256->12*12*384 (12-3)/1+1=12
            nin_block(256,384,kernel_size=3,stride=1,padding=1),
            # 12*12*384->5*5*384(12-3)/2+1=5
            nn.MaxPool2d(kernel_size=3,stride=2),nn.Dropout(0.5),
            # 5*5*384->5*5*10 (5-3+1*2)/1+1=5
            nin_block(384,10,kernel_size=3,stride=1,padding=1),
            # 自适应平均池化，会将长度设置为1  1*1*310
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),

        )

    def forward(self,x):
        x=self.model(x)
        return x


def test():
    x=torch.randn(1,1,224,224)
    model=NiN()
    for layer in model.model:
        x=layer(x)
        print(x)
        print(layer.__class__.__name__,x.shape)

# test()
