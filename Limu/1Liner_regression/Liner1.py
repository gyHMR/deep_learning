import numpy as np
import torch
from torch.utils import data
from torch import nn

true_w=torch.tensor([2,-3.4])
true_b=4.2

# 生成数据
def synthetic_data(w, b, num_examples):
    # x=torch.normal(mead=0,std=1,size=(num_examples,len(w)))  #均值为0，标准差为1，大小为num_examples行，w列
    x = torch.tensor(np.random.normal(0, 1, size=(num_examples, len(w))), dtype=torch.float32)
    y = torch.matmul(x, w) + b
    # 添加一个噪声
    # y+=torch.normal(mean=0,std=0.01,size=y.shape)
    y += torch.tensor(np.random.normal(0, 0.01, size=y.size()), dtype=torch.float32)
    # 将y由num_examples行变为num_examples行，1列
    return x, y.reshape((-1, 1))
features,lables=synthetic_data(true_w,true_b,1000)


# 设置超参数
epochs=3
batch_size=10
lr=0.03

# 得到批量数据
def data_load1(data_array,batch_size,shuffle=True):
    data_set=data.TensorDataset(*data_array)
    return data.DataLoader(data_set,batch_size,shuffle)
data_iter=data_load1((features,lables),batch_size)
next(iter(data_iter))


#线性回归模型
net=nn.Sequential(nn.Linear(2,1))  #输入2，输出1
# 初始化参数
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

# 误差函数
loss=nn.MSELoss()
# 优化器
trainer=torch.optim.SGD(net.parameters(),lr)

# 开始训练
for epoch in range(epochs):
    for x,y in data_iter:
        l=loss(net(x),y)  #计算损失函数
        trainer.zero_grad() #清空梯度
        l.backward()   #反向传播
        trainer.step() #更新参数
    with torch.no_grad():
        l=loss(net(features),lables)
        print(f'epochs{epoch}:loss{l:f}')


