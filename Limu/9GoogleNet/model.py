import torch
from torch import nn


# 基础卷积模板，每次卷积后都要跟上一次非线性激活
class basic_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(basic_conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.relu=nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x



# inecption块结构
class Inception(nn.Module):
    '''
    ch1x1:第一个分支1x1卷积的通道数
    ch3x3_1:第二个分支上1x1卷积的通道数
    ch3x3_2:第二个分支上3x3卷积的通道数
    ch5x5_1：第三个分支上1x1卷积的通道数
    ch5x5_2：第三个分支上5x5卷积的通道数
    ch1x1_pool：第四个分支上经过max_pool之后的那个1x1卷积的通道数
    '''
    def __init__(self, in_channels, ch1x1,ch3x3_1,ch3x3_2,ch5x5_1,ch5x5_2,ch1x1_pool):
        super(Inception, self).__init__()
        self.branch1=basic_conv2d(in_channels,ch1x1,kernel_size=1)
        self.branch2=nn.Sequential(
            basic_conv2d(in_channels,ch3x3_1,kernel_size=1),
            basic_conv2d(ch3x3_1,ch3x3_2,kernel_size=3,padding=1), #加上pad保证大小不变
        )
        self.branch3=nn.Sequential(
            basic_conv2d(in_channels,ch5x5_1,kernel_size=1),
            basic_conv2d(ch5x5_1,ch5x5_2,kernel_size=5,padding=2),#加上pad保证大小不变
        )
        self.branch4=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),   #maxpoll不改变通道数
            basic_conv2d(in_channels,ch1x1_pool,kernel_size=1),
        )

    def forward(self,x):
        branch1=self.branch1(x)
        branch2=self.branch2(x)
        branch3=self.branch3(x)
        branch4=self.branch4(x)
        # 沿着第二个维度（通道数）将张量拼接起来
        return torch.cat((branch1,branch2,branch3,branch4),dim=1)


# 辅助分类器
class Inception_aux(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(Inception_aux,self).__init__()
        self.moelf=nn.Sequential(
            nn.AvgPool2d(kernel_size=5,stride=3),
            basic_conv2d(in_channels,128,kernel_size=1),
            nn.Flatten(),
            nn.Dropout(0.5),
            #两个全连接
            nn.Linear(128*4*4,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,num_classes),
        )

    def forward(self,x):
        x=self.moelf(x)
        return x


# Googlenet网络
class Googlenet(nn.Module):
    def __init__(self,num_classes=1000,aux_logits=True,):
        super(Googlenet,self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = basic_conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)  # ceil_mode=True 计算为小数时，向上取整
        self.conv2 = basic_conv2d(64, 64, kernel_size=1)
        self.conv3 = basic_conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        # 辅助分类器
        if aux_logits:
            self.aux1 = Inception_aux(512, num_classes)
            self.aux2 = Inception_aux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:  # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:  # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:  # eval model lose this layer
            return x, aux2, aux1
        return x


def test():
    x = torch.randn(1, 3, 224, 224)
    model = Googlenet()
    model.eval()
    output = model(x)
    print("Output shape of GoogLeNet:", output.shape)


test()
