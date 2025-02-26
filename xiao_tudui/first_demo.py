# #读取数据
# import os
# from torch.utils.data import  dataset
# from PIL import Image
#
# class Mydata(dataset):
#     def __init__(self,rootdir,labeldir):
#         self.rootdir=rootdir
#         self.labeldir=labeldir
#         self.path=os.path.join(rootdir,labeldir)
#         self.img_path=os.listdir(self.path)
#
#     def __getitem__(self, idx):
#         img_name=self.img_path[idx]
#         img_item_path=os.path.join(self.rootdir,self.labeldir,img_name)
#         img=Image.open(img_item_path)
#         label=self.labeldir
#         return  img,label
#
#     def __len__(self):
#         return len(self.img_path)
#
# rootdir="dataset/hymenoptera_data/train"
# ants_label_dir="ants"
# ans_dataset=Mydata(rootdir,ants_label_dir)


##### tensorbord使用练习
# import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# import  cv2
# from  PIL import Image

# # 写表格
# writer=SummaryWriter("logs")
# for i in range(100):
#     writer.add_scalar("y=x^2",i**2,i)

# # 写图片
# writer=SummaryWriter("logs")
# img_path="dataset/hymenoptera_data/train/ants/0013035.jpg"
# img=cv2.imread(img_path)
# print(type(img))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB
# img = img.astype('float32') / 255.0  # 归一化到 [0, 1]
#
# writer.add_image("test",img,1)


# 卷积操作的尝试
import  torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 获取数据集
dataset=torchvision.datasets.CIFAR10(root="./data",train=False,transform=torchvision.transforms.ToTensor()
                                     ,download=True)
# 装载数据集
dataloader=DataLoader(dataset,batch_size=64)

# 设置模型
class Mymodel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),  # 第一次卷积
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64, 10)
        )
        # self.conv1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2) #第一次卷积
        # self.maxpoll1=nn.MaxPool2d(kernel_size=2)
        # self.conv2=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2)
        # self.maxpoll2=nn.MaxPool2d(kernel_size=2)
        # self.conv3=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2)
        # self.maxpoll3=nn.MaxPool2d(kernel_size=2)
        # self.flatten=nn.Flatten()
        # self.linear1=nn.Linear(1024,64)
        # self.linear2=nn.Linear(64,10)


    def forward(self,x):
        x=self.module(x)
        return x



mymodel=Mymodel()
step=0
writer=SummaryWriter("./logs")
for data in dataloader:
    imgs,targets=data
    print(imgs.shape)
    imgs_squeezed = imgs.squeeze(0)
    writer.add_image("input",imgs_squeezed,step,dataformats="CHW")
    output=mymodel(imgs)  #卷积操作
    writer.add_image('output', output, step)
    print(output.shape)
    step=step+1

