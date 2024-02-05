import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

k = 1000
a = 1*10**(-1)
w = 100
b = 150
dataset = pd.read_csv("./dataset/archive/brca.csv")
m = dataset.shape[0]
x = dataset["x.radius_mean"]
x_plt = dataset["ID"]
ordinal_encoder = OrdinalEncoder()
y = pd.DataFrame(dataset["y"])
y = ordinal_encoder.fit_transform(y)




def sigmoid(n):
    return 1/(1+np.exp(-n))

def gradient(w,b, m):
    dj_dw = 0
    dj_db = 0
    z = w*x+b
    f_iw = sigmoid(z)
    for i in range(m):
        dj_dw += (1/m)*(f_iw[i]-y[i][0])*x[i]
        dj_db += (1/m)*(f_iw[i]-y[i][0])
    return dj_dw, dj_db

wl, bl = [],[]
def gradient_descent(w,b,m,a):

    dj_dw , dj_db = gradient(w,b,m)
    tm_w = w - a*dj_dw
    tm_b = b - a*dj_db
    w = tm_w
    b = tm_b
    return w,b, wl,bl

def compute(x,w,b,m, k,wl,bl):
    tmp_plt = np.zeros(m)
    i = 0
    while (i < k):
        w, b, wl, bl= gradient_descent(w, b, m, a)
        wl.append(w)
        bl.append(b)
        i += 1
    for i in range(m):
        tmp_plt[i] = 1/(1+np.exp(w*x[i]+b))
    return tmp_plt,w,b


prediction,w,b = compute(x,w,b,m,k,wl,bl)

x = np.linspace(0, 50, 100)
y = 1/(1+np.exp(w*x+b))
topred = float(input("Enter the radius of the tumour: "))
lr = 1/(1+np.exp(w*topred+b))
if (lr>=0.5):
    print("Maliglant")
else:
    print("Benign")
plt.plot(x,y)
plt.scatter(topred,lr, marker="^", c="r")
x = np.linspace(0,50,100)
y = np.repeat(0.5,100)
plt.plot(x,y,"g--")
plt.show()
"""
"""
