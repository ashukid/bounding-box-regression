import cv2 as cvision
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib import patches
get_ipython().run_line_magic('matplotlib', 'inline')



class Data:
    def __init__(self,root,csv,batch_size=32):
        self.root=root
        self.csv=pd.read_csv(csv)
        self.batch_size=batch_size
        self.idx=0
        self.image_size = self.set_image_size()
        
        
    def get_next_batch(self):
        x=[]
        y=[]
        tempy=self.csv.iloc[self.idx:self.idx+self.batch_size,:].values
        for row in tempy:
            y.append(self.normalize(row[1:]))
            tempx=cvision.imread(os.path.join(self.root,row[0]))
            tempx=np.array(np.moveaxis(tempx,-1,0),dtype=np.float32)
            x.append(tempx)
            
        self.idx=(self.idx+self.batch_size)%len(self.csv)
        return np.array(x),np.array(y)
    
    
    def get_test_data(self):
        x=[]
        tempy=self.csv.iloc[self.idx:self.idx+self.batch_size,0].values
        for row in tempy:
            tempx=cvision.imread(os.path.join(self.root,row))
            tempx=np.array(np.moveaxis(tempx,-1,0),dtype=np.float32)
            x.append(tempx)
        self.idx=(self.idx+self.batch_size)%len(self.csv)
        return np.array(x)
    
    def set_image_size(self):
        im=self.csv.iloc[0,0]
        im=cvision.imread(os.path.join(self.root,im))
        return im.shape
    
    def normalize(self,arr):
        arr[0] /= self.image_size[1]
        arr[1] /= self.image_size[1]
        arr[2] /= self.image_size[0]
        arr[3] /= self.image_size[0]
        return np.array(arr,dtype=np.float32)
        



train_data=Data('train','training.csv',32)
train_data.image_size



class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,8,5,1,2)
        self.conv2=nn.Conv2d(8,16,5,1,2)
        self.conv3=nn.Conv2d(16,32,5,1,2)
        self.conv4=nn.Conv2d(32,64,5,1,2)

        self.fc1=nn.Linear(64*30*40,512)
        self.fc2=nn.Linear(512,4)
    
    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),2)
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=F.max_pool2d(F.relu(self.conv3(x)),2)
        x=F.max_pool2d(F.relu(self.conv4(x)),2)
        
        x=x.view(-1,64*30*40)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        
        return x
        
net=Net()


params=list(net.parameters())
params[1]


criterion=nn.MSELoss()
optim=Adam(net.parameters(),lr=0.001)



for epoch in range(1):
    
    for i in range(len(train_data.csv)//train_data.batch_size):
        
        x,y=train_data.get_next_batch()
        x,y=Variable(torch.from_numpy(x)),Variable(torch.from_numpy(y))
        optim.zero_grad()        
        out=net(x)
        loss=criterion(out,y)
        loss.backward()
        optim.step()
        
        if(i%100==0):
            print("Epoch :{} Iter : {} Loss : {} ".format(epoch,i,loss))



test_csv=pd.read_csv('test.csv')
test_data=Data('test','test.csv')

net.eval()
y=[]
for i in range((len(test_data.csv)//test_data.batch_size)+1):
    x=test_data.get_test_data()
    print(i,x.shape)
    x=Variable(torch.from_numpy(x))
    try:
        out=net(x).detach().numpy()
        y.extend(out)
    except exception as e:
        print(e)


y=np.array(y)

pred=y[:]

def denormalize(a):
    
    for arr in a:
        arr[0]*=test_data.image_size[1]
        arr[1]*=test_data.image_size[1]
        arr[2]*=test_data.image_size[0]
        arr[3]*=test_data.image_size[0]
    return a
pred=denormalize(pred)
test_csv.iloc[:,2]=pred[:,1]
test_csv.iloc[:,3]=pred[:,2]
test_csv.iloc[:,4]=pred[:,3]
test_csv.to_csv('test.csv')




