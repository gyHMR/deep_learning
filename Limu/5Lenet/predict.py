import torch
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from LeNet import leNet


my_transform=transforms.Compose([transforms.ToTensor()])
train_data=datasets.MNIST(root="./data",train=True,transform=my_transform,download=False)
test_data=datasets.MNIST(root="./data",train=False,transform=my_transform,download=False)

train_loader=DataLoader(train_data,batch_size=16,shuffle=True)
test_loader=DataLoader(test_data,batch_size=16,shuffle=True)

device="cuda" if torch.cuda.is_available() else "cpu"
net=leNet().to(device)
# 加载模型
net.load_state_dict(torch.load("./save_model/best_model.pth"))

# 开始评估模型
net.eval()
sum_loss,sum_acc,n=0,0,0
with torch.no_grad():
    for batch,(x,y) in enumerate(test_loader):
        x,y=x.to(device),y.to(device)
        output=net(x)
        _,pred=torch.max(output,dim=1)
        acc = torch.sum(pred == y).item() / y.shape[0]
        n+=1
        sum_acc+=acc
    print(f"准确率为：{sum_acc/n}")







