

import matplotlib.pyplot as plt
import numpy as np

x_train = np.arange(0.,np.pi*2,0.1)
y_train = np.sin(x_train)
# min max normalization
x_train = (x_train-np.pi)/np.pi
m = len(x_train)

class MiddleLayer:
    def __init__(self,matM,matN):
        self.w = np.random.randn(matM,matN)
        self.b = np.random.randn(matN)
    def forward(self,x):
        self.x = x
        self.z = x@self.w+self.b
        self.s = 1/(1+np.exp(-self.z))
        
    def backward(self,grad):
        delta = grad*(1-self.s)*self.s
        self.dw = self.x.T@delta
        self.db = np.sum(delta,axis=0)
    def update(self,eta):
        self.w -= eta*self.dw
        self.b -= eta*self.db

class OutputLayer:
    def __init__(self,matM,matN):
        self.w = np.random.randn(matM,matN)
        self.b = np.random.randn(matN)
    def forward(self,x):
        self.x = x
        self.z = x@self.w + self.b
        self.y = self.z # 회귀의 경우 항등함수
        
    def backward(self,t):

        self.dy = self.y-t

        self.dw = self.x.T @ self.dy 
        self.db = np.sum(self.dy,axis=0)
        self.dx = self.dy @ self.w.T
    def update(self,eta):
        self.w -= eta*self.dw
        self.b -= eta*self.db


middle_layer = MiddleLayer(1,3)
output_layer = OutputLayer(3,1)

epoch = 100
eta = 0.3

for j in range(epoch+1):
    index_random = np.arange(m)
    np.random.shuffle(index_random)
    
    plot_x = []
    plot_y = []
    total_error = 0
    for i in index_random:
        x = x_train[i:i+1]
        t = y_train[i:i+1]

        
        middle_layer.forward(x.reshape(1,1))
        output_layer.forward(middle_layer.s)
        
        output_layer.backward(t.reshape(1,1))
        middle_layer.backward(output_layer.dx)
        output_layer.update(eta)
        middle_layer.update(eta)
        
        if j%100==0:
            plot_x.append(x)
            plot_y.append(output_layer.y)
            total_error += (1/2)*np.sum(np.square(output_layer.y.reshape(-1)-t))
plt.plot(x_train,y_train)
plt.scatter(plot_x,plot_y,marker="x")
plt.show()

print()