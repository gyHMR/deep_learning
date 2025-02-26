import random
import torch
import matplotlib.pyplot as plt
import numpy as np

# 随机数据生成函数
def synthetic_data(w,b,num_examples):
     # x=torch.normal(mead=0,std=1,size=(num_examples,len(w)))  #均值为0，标准差为1，大小为num_examples行，w列
     x= torch.tensor(np.random.normal(0, 1, size=(num_examples, len(w))), dtype=torch.float32)
     y=torch.matmul(x,w)+b
     #添加一个噪声
     # y+=torch.normal(mean=0,std=0.01,size=y.shape)
     y+= torch.tensor(np.random.normal(0, 0.01, size=y.size()), dtype=torch.float32)
     # 将y由num_examples行变为num_examples行，1列
     return x,y.reshape((-1,1))

# 生成数据集
true_w=torch.tensor([2,-3.4])
true_b=4.2
features,lables=synthetic_data(true_w,true_b,1000)

# # 绘图展示数据
# plt.figure(figsize=(10,6))
# plt.scatter(features[:,1].detach().numpy(),lables.detach().numpy(),1)
# plt.title("liner")
# plt.xlabel("feature")
# plt.ylabel("label")
# plt.show()

# 生成小批量数据
def data_iter(batch_size,features,lables):
     num=len(features)
     index=list(range(num)) #生成一个索引列表
     random.shuffle(index)  # 随机打乱索引列表
     for i in range(0,num,batch_size):
          j=torch.tensor(index[i:min(i+batch_size,num)])
          yield features[j],lables[j]

# 定义模型参数
w=torch.normal(mean=0,std=0.01,size=(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)

# 线性回归模型定义
def linier(x,w,b):
     return torch.matmul(x,w)+b

# 损失函数
def loss(y_hat,y):
     return (y_hat-y.reshape(y_hat.shape))**2/2

# 优化算法
def sgd(params,lr,batch_size):
     with torch.no_grad():
          for param in params:
               param -= lr*param.grad/batch_size   #减去学习率乘以梯度除以批量大小
               param.grad.zero_()

# 设置超参数
lr=0.03
epochs=3
batch_size=10

# 对数据扫epochs次
for epoch in range(epochs):
     # 每次拿出一个小批量
     for x,y in data_iter(batch_size,features,lables):
          l=loss(linier(x,w,b),y)  #计算损失函数
          l.sum().backward()  #因为是线性的，故直接计算一个批量的损失和、然后反向传播
          sgd([w,b],lr,batch_size)  #优化器
     # 扫完一次后计算预测和真实值
     with torch.no_grad():
          train_l=loss(linier(features,w,b),lables)
          print(f'epoch{epoch}: loss{float(train_l.mean()):f}')  # :f意义为格式化（只显示六位小数）



