import numpy as np
import matplotlib.pyplot as plt
import math
import tqdm
import pandas  as pd
import GetMatrix as GM

N = 2

def Move(frm, to):
    for i in range(to.size):
        to[i] = frm[i]
    return True

def f(x, A, b):
    res = 2 * (np.dot(A, x) - b).T.dot(A)
    return res


def GradientDescent(x_0, iteration):
    r_x = np.zeros((N, 1))
    for i in range(N):
        r_x[i][0] = x_0[i]
    
    gamma = 0.001
    
    #A, b = GM.ReadData()
    
    theta1 = np.zeros(100)
    theta2 = np.zeros(100)
    
    for i in range(100):
        theta1[i] = 2*i
        theta2[i] = 3*i - 20
    
    delta = np.random.uniform(-10,10, size=(100,))
    A = np.stack((theta1, theta2)).T
    b = 0.4 * theta1 - 1.3 * theta2 + delta
    #print(A)
    
    x = x_0 - gamma * f(x_0, A, b)
    

    
    x_l = x_0
    for i in tqdm.tqdm(range(iteration)):
        if np.sqrt(np.sum(x**2)) < 10**(-30):
            for i in range(x.size):
                x[i] = 0
        else:
            if np.dot(f(x, A, b)-f(x_l, A, b), f(x, A, b)-f(x_l, A, b)) < 10**(-20):
                gamma = 10**(-20)
            else:
                gamma = np.dot(x-x_l, f(x, A, b)-f(x_l, A, b)) / np.dot(f(x, A, b)-f(x_l, A, b), f(x, A, b)-f(x_l, A, b))
            
            #print(np.dot(f(x, A, b)-f(x_l, A, b), f(x, A, b)-f(x_l, A, b)))
            #print(gamma)
            #print('_____________________')
            x_n = x - gamma * f(x, A, b)
            Move(x, x_l)
            Move(x_n, x)
        
        #print(x)
        r_x = np.append(r_x, x.reshape((N,1)), axis=1)
    
    #plt.plot(r_x[0, :], r_x[1, :])
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