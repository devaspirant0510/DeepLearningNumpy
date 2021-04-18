import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
x_train = np.arange(-1.0,1.1,0.1)
y_train = np.arange(-1.0,1.1,0.1)
path = os.getcwd()+"\\example"
train_img_path = path+"\\train_data.png"
def draw_train_data():
    plt.plot(x_train,np.sin(np.pi*x_train),'r')
    plt.title("train_data")
    plt.xlabel("x")
    plt.ylabel("sin(pi*x)")
    plt.savefig(train_img_path)
#draw_train_data()
input_data = []
output_data = []
for i in x_train:
    for j in y_train:
        input_data.append([i,j])
        if j<np.sin(np.pi*i):
            output_data.append([0,1])
        else:
            output_data.append([1,0])
input_data = np.array(input_data)
output_data = np.array(output_data)
class MiddleLayer:
    def __init__(self,Wm,Wn):
        self.w = np.random.randn(Wm,Wn)
        self.b = np.random.randn(Wn)
    def forward(self,x):
        self.x = x
        self.z = x@self.w+self.b
        self.s = 1/(1+np.exp(-self.z))
    def backward(self,grad):
        delta =   grad *self.s * (1-self.s)
        self.dw = self.x.T @ delta
        self.db = np.sum(delta,axis=0)
        self.dx = delta @ self.w.T 
    def update(self,lr):
        self.w -= lr*self.dw
        self.b -= lr*self.db

class OutputLayer:
    def __init__(self,Wm,Wn):
        self.w = np.random.randn(Wm,Wn)
        self.b = np.random.randn(Wn)
    def forward(self,x):
        l = 1e-5
        self.x = x
        self.z = x@self.w+self.b
        self.softmax = np.exp(self.z)/np.sum(np.exp(self.z),axis=1,keepdims=True)
        self.loss = (1/len(output_data))*-np.sum((1-output_data)*np.log(1-self.softmax+l)+output_data*np.log(self.softmax+l))
    def backward(self,grad):
        self.dact = grad-output_data
        self.dw = self.x.T @ self.dact
        self.db = np.sum(self.dact,axis=0)
        self.dx = self.dact@self.w.T
    def update(self,lr):
        self.w -= lr*self.dw
        self.b -= lr*self.db


hiddenLayer = MiddleLayer(2,10)
ouputlayer = OutputLayer(10,2)
sindata = np.sin(np.pi*x_train)
epoch = 10001
eta = 0.001
ndata = len(output_data)
for i in range(epoch):
    indexRandom = np.arange(ndata)
    np.random.shuffle(indexRandom)

    total_error = 0
    hiddenLayer.forward(input_data)
    ouputlayer.forward(hiddenLayer.s)
    ouputlayer.backward(ouputlayer.softmax)
    hiddenLayer.backward(ouputlayer.dx)
    ouputlayer.update(eta)
    hiddenLayer.update(eta)
    y = ouputlayer.softmax.reshape(-1)
    total_error+=ouputlayer.loss
    # if(ouputlayer.loss<1):
    #     break
    if i%100==0:
        print(ouputlayer.loss)
    # for j in indexRandom:
    #     x = input_data[j]
    #     t = output_data[j]
    #     hiddenLayer.forward(x.reshape(-1,2))
    #     ouputlayer.forward(hiddenLayer.s)
    #     ouputlayer.backward(t.reshape(-1,2))
    #     hiddenLayer.backward(ouputlayer.dx)
    #     ouputlayer.update(eta)
    #     hiddenLayer.update(eta)
    #     if (i%100==0):
    #         y = ouputlayer.softmax.reshape(-1)
    #         total_error+=ouputlayer.loss
    #         if y[0]>y[1]:
    #             x_1.append(x[0])
    #             y_1.append(x[1])
    #         else:
    #             x_2.append(x[0])
    #             y_2.append(x[1])
    # if (i%10==0):
    #     plt.plot(x_train,sindata,"--")
    #     plt.scatter(x_1,y_1,marker="+")
    #     plt.scatter(x_2,y_2,marker="o")
    #     plt.show()




y_data = np.sin(np.pi*x_train)
predict = ouputlayer.softmax
x_1 = []
x_2 = []
y_1 = []
y_2 = []
for idx,i in enumerate(predict):
    if i[0]<i[1]:
        x_1.append(input_data[idx,0])
        y_1.append(input_data[idx,1])
    else:
        x_2.append(input_data[idx,0])
        y_2.append(input_data[idx,1])
plt.plot(x_train,y_data)
plt.scatter(x_1,y_1,marker="o")
plt.scatter(x_2,y_2,marker="x")
plt.show()