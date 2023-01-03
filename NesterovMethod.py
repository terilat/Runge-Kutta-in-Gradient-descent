import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import *
import GetMatrix as GM

N = 2

def f(x, A, b):
    return 2 * (np.dot(A, x) - b).T.dot(A)

def NesterovMethod(x_0, h, iteration):
    r_x = np.copy(x_0)
    r_x = r_x.reshape((N, 1))
    #A, b = GM.ReadData()
    theta1 = np.zeros(100)
    theta2 = np.zeros(100)
    
    for i in range(100):
        theta1[i] = 2*i
        theta2[i] = 3*i - 20
    
    delta = np.random.uniform(-10,10, size=(100,))
    A = np.stack((theta1, theta2)).T
    b = 0.4 * theta1 - 1.3 * theta2 + delta
    
    
    x = x_0 - h * f(x_0, A, b)
    y = np.copy(x)
    
    for i in tqdm(range(2, iteration)):
        if np.sqrt(np.sum(x**2)) < 10**(-30):
            for i in range(x.size):
                x[i] = 0
        else:
            x_n = y - h * f(y, A, b)
            y = x + (i - 1)/(i + 2) * (x_n-x)
            x = np.copy(x_n)
        #print(x)
        r_x = np.append(r_x, x.reshape((N,1)), axis=1)
    
    #plt.plot(r_x[0, :], r_x[1, :])
    #plt.show()
    #norm_distr = np.sqrt(r_x[0, :]**2 + r_x[1, :]**2)
    #plt.loglog(norm_distr)
    #plt.show()
    
    print(x)
    
    print(GM.Error(A, b, x))
    #x = np.array([257.28807485806226, -131.11057962209236, 332.57013224229627, 479.36939354512685, -23820.434122672905, 353.6400165588405, 10944.061751765028])
    x = np.array([0.4, -1.3])
    print(GM.Error(A, b, x))
    
    norm_distr = np.zeros(r_x.shape[1])
    for i in range(r_x.shape[1]):
        norm_distr[i] = np.sqrt((A.dot(r_x[:, i]) - b).dot(A.dot(r_x[:, i]) - b))
    plt.plot(norm_distr)
    plt.show()
    
    
    return norm_distr