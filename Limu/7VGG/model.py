import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader


class Vgg(nn.Module):
    def __init__(self):
        super(Vgg,self).__init__()
        # 实际输入的32*32*3的数据
        # 第一块
        self.layer1=nn.Sequential(
            # 224*2224*3-》224*224*64  (224-3+1*2)/1+1=224
            nn.Conv2d(3,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 224*224*64->224*224*64
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 224*224*64->112*112*64   (224-2)/2+1=112
            nn.MaxPool2d(2,2)
        )

        #第二块
        self.layer2=nn.Sequential(
            #112*112*64->112*112*128
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #112*112*12->112*112*128
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #112*112*128->56*56*128  (112-2)/2+1=56
            nn.MaxPool2d(2,2)
        )

        # 第三块
        self.layer3 = nn.Sequential(
            # 56*56*128 ->56*56*256
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 156*56*256->56*56*256
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 156*56*256->56*56*256
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 56*56*256->28*28*256  (56-2)/2+1=28
            nn.MaxPool2d(2, 2)
        )

        # 第四块
        self.layer4 = nn.Sequential(
            # 28*28*256->28*28*512
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 28*28*512->28*28*512
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 28*28*512->28*28*512
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 28*28*512->14*14*512 (28-2)/2+1=14
            nn.MaxPool2d(2, 2)
        )

        # 第五块
        self.layer5 = nn.Sequential(
            # 14*14*512 ->14*14*512
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 14*14*512 ->14*14*512
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 14*14*512 ->14*14*512
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 14*14*512 ->7*7*512 (14-2)/2+1=7
            nn.MaxPool2d(2, 2)
        )
        # 方便调用
        self.features=nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            # 摊平 以便后面的全连接
            nn.Flatten()
        )

        #全连接层
        self.fc=nn.Sequential(
            nn.Linear(512,4096),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        # 分类
        self.classifier=nn.Sequential(
            # 实际上是一个十分类问题
            nn.Linear(4096,10)
        )


    def forward(self,x):
        x=self.features(x)
        x=self.fc(x)
        x=self.classifier(x)
        return x


def test():
    net = Vgg()
    x = torch.randn(1, 3, 32, 32)

    # 测试卷积块
    print("Features Block:")
    features = [
        net.layer1, net.layer2, net.layer3, net.layer4, net.layer5
    ]
    for layer in features:
        x = layer(x)
        print(f"{layer.__class__.__name__} Output Shape: {x.shape}")

    # 测试全连接层
    print("\nFC Block:")
    x = torch.flatten(x, 1)  # 展平特征图
    fc_layers = [net.fc[0], net.fc[1], net.fc[2], net.fc[3], net.fc[4]]
    for layer in fc_layers:
        x = layer(x)
        print(f"{layer.__class__.__name__} Output Shape: {x.shape}")

    # 测试分类器
    print("\nClassifier Block:")
    x = net.classifier(x)
    print(f"{net.classifier.__class__.__name__} Output Shape: {x.shape}")

# test()