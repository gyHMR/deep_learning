import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from model import NiN


my_transforms=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
])

test_data=datasets.CIFAR10(root="./data",train=False,transform=transforms.ToTensor(),download=False)

test_dataloader=DataLoader(dataset=test_data,batch_size=32,shuffle=True)

device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
net=NiN().to(device)
net.load_state_dict(torch.load("./save_model/best_model.pth"))

# 开始评估模型
net.eval()
sum_loss,sum_acc,n=0,0,0
with torch.no_grad():
    for batch,(x,y) in enumerate(test_dataloader):
        x,y=x.to(device),y.to(device)
        output=net(x)
        _,pred=torch.max(output,dim=1)
        acc = torch.sum(pred == y).item() / y.shape[0]
        n+=1
        sum_acc+=acc
    print(f"准确率为：{sum_acc/n}")