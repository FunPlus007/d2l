import torch
import numpy as np
from torch.utils import data
from d2l import torch as d2l

print("hello torch")

true_w = torch.tensor([2,-3.5])
true_b = 4.2

# make data set 
features ,labels = d2l.synthetic_data(true_w, true_b,1000)

def load_array(data_arrays,batch_size,is_trian = True):
    #@save
    '''PyTorch data 迭代器'''
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_trian)

batch_size = 10
data_iter = load_array((features,labels),batch_size,is_trian=True)

next(iter(data_iter))
print(next(iter(data_iter)))


# 使用torch中封装好的模型框架
from torch import nn

net = nn.Sequential(nn.Linear(2,1))

net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(),lr = 0.03)

num_epochs = 10
for epoch in range(num_epochs):
    for X , y in data_iter:
        l = loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss (net(features),labels)
    print(f'epoch:{epoch},loss:{l:f}')


w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)