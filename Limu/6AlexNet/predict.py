import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from AlexNet import  alexnet

my_transform=transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ]
)

# 下载数据
test_data=datasets.FashionMNIST(root="./data",train=False,transform=transforms.ToTensor(),download=False)

# 加载数据
test_loader =DataLoader(dataset=test_data,batch_size=64,shuffle=True)

device="cuda" if torch.cuda.is_available() else "cpu"
net=alexnet().to(device)
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