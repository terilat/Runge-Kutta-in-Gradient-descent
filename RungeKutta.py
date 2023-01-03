import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import *
import GetMatrix as GM
import copy
a = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1.0/5, 0, 0, 0, 0, 0, 0],
              [0, 3.0/40, 9.0/40, 0, 0, 0, 0, 0],
              [0, 44.0/45, -56.0/15, 32.0/9, 0, 0, 0, 0],
              [0, 19372.0/6561, -25360.0/2187, 64448.0/6561, -212.0/729, 0, 0, 0],
              [0, 9017.0/3168, -355.0/33, 46732.0/5247, 49.0/176, -5103.0/18656, 0, 0],
              [0, 35.0/384, 0, 500.0/1113, 125.0/192, -2187.0/6784, 11.0/84, 0]])
             
b = np.array([0, 35.0/384, 0, 500.0/1113, 125.0/192, -2187.0/6784, 11.0/84, 0])
b_k = np.array([0, 5179.0/57600, 0, 7571.0/16695, 393.0/640, -92097.0/339200, 187.0/2100, 1.0/40])
c = np.array([0, 0, 1.0/5, 3.0/10, 4.0/5, 8.0/9, 1.0, 1.0])

P = 5 #Порядок метода
N = 4 #Размерность системы уравнений
n = 2
eps = 10**(-8)  #Погрешность вычислений

             
#Норма для метода Рунге-Кутта
def Norma(x, y):         
    return np.abs(x - y).max() / (pow(2, P)-1)

#Обычная евклидова норма
def UsualNorma(x):
    return np.sqrt(np.sum(x**2))
    
#Векторное поле в точке x[]

def f(t, x, res, q, A, b):
    res[:n] = -(2*q+1)/t * x[:n] - 2*q**2 * math.pow(t, q-2) * (np.dot(A, x[n:]) - b).T.dot(A)[:n]
    res[n:] = x[:n]
    #res[0] = -(2*q+1)/t * x[0] - 2*q**2 * math.pow(t, q-2) * (np.dot(A, x[N/2:]) - b).T.dot(A)[0]
    #res[1] = -(2*q+1)/t * x[1] - 2*q**2 * math.pow(t, q-2) * (np.dot(A, x[N/2:]) - b).T.dot(A)[1]
    #res[2] = x[0]
    #res[3] = x[1]
    return True


#Копирование элементов массива from в массив to
def Move(frm, to):
    for i in range(to.shape[0]):
        to[i] = frm[i]
    return True

#Метод Рунге-Кутта
def RungeKutta(h, x_0, t_0, T, iteration, q, filename = 'NAN', tol = 10**(-12)):
    r_x = np.zeros((N, 1))
    for i in range(N):
        r_x[i][0] = x_0[i]
    x = np.zeros(N)
    dx = np.zeros(N)
    t = t_0
    x_l = np.zeros(N) #хранит значения x на предыдущей итерации
    tmp_x = np.zeros(N)
    res = np.zeros(N) #tmp_x - для вычисления промежуточных значений, res[N] - массив, для хранения компонент векторного поля в точке
    kx = np.zeros((8, N)) #коэффициенты k при вычислении сумм в методе Рунге-Кутта
    x_k = np.zeros(N)
    x_p = np.zeros(N) #x_k[N] - значения с крышечкой, x_p[N] - переменные для суммирования
    IsOpen = False   #IsOpen - флаг открытия файла. если файл открыт, то запись в него идет промежуточных значений, если нет, то не идет)))
    fac = 0.9
    facmax = 1.5 
    facmin = 0.7 #параметры для автошага
             
    delta = 0 #Переменная для вычисления погрешности
    steps = 0 #число шагов
    error = 0 #число для отслеживания зацикливания
    
    t_n = 0
    
    H = 100*h
    
    x = np.copy(x_0)
    x_l = np.copy(x_0)
    #Move(x_0, x)
    #Move(x_0, x_l)
    
    if filename != 'NAN':
        fout = fopen(filename, 'a')
        IsOpen = True
    
    #data, y = GM.ReadData()
    theta1 = np.zeros(100)
    theta2 = np.zeros(100)
    
    for i in range(100):
        theta1[i] = 2*i
        theta2[i] = 3*i - 20
    
    delta = np.random.uniform(-10,10, size=(100,))
    data = np.stack((theta1, theta2)).T
    y = 0.4 * theta1 - 1.3 * theta2 + delta
    #while t < T and steps < 10000:
    for steps in tqdm(range(iteration)):
        #чтобы последняя точка была вычислена ровно в последний момент времени
        #if t+h > T: 
            #h = T - t

        #вычисление сумм аргументов
        for i in range(1, 8):
            Move(x, tmp_x)
            for j in range(1, i):
                for l in range(N):
                    tmp_x[l] += h*a[i][j]*kx[j][l]

            f(t + c[i]*h, tmp_x, res, q, data, y)
            for l in range(N):
                kx[i][l] = res[l]
        
        x_k = np.copy(x)
        x_p = np.copy(x)
        #Move(x, x_k)
        #Move(x, x_p)
             
        for i in range(1, 8):
            for l in range(N):
                x_p[l] += h*b[i]*kx[i][l]
                x_k[l] += h*b_k[i]*kx[i][l]
            
        
        #условие на то, допустима ли точность следующего шага
        
        if Norma(x_p, x_k) < tol:
            x_l = np.copy(x)
            x = np.copy(x_p)
            #Move(x, x_l)
            #Move(x_p, x)
            #print(r_x.shape)
            #print(x.shape)
            r_x = np.append(r_x, x.reshape((N,1)), axis=1)
            t += h
            steps += 1

            if IsOpen:
                fout.write(str(t) + ' ' + str(x[0]) + ' ' + str(x[1]) + ' ' + str(x[2]) + ' ' + str(x[3]) + '\n')
            delta += Norma(x_p, x_k)
        
        #автошаг
        if Norma(x, x_k) < tol**3:
            h = h * min(facmax, facmin)
            #print(steps, Norma(x, x_k))
            #print(x)
            #print(x_k)
            #print(h)
            #print('_______________________')
            
        else:
            h = h * min(facmax, max(facmin, fac * math.pow(tol / Norma(x, x_k), 1.0/(P + 1))))
        if h > H:
            h = copy.deepcopy(H)
        #print(steps, h)
        error += 1
        steps -= 1
        
    if IsOpen:
        fout.close()
    print(t)
    #plt.plot(r_x[0, :], r_x[1, :])
    #plt.show()
    
    print(x)
    
    print(GM.Error(data, y, x[:n]))
    #x = np.array([257.28807485806226, -131.11057962209236, 332.57013224229627, 479.36939354512685, -23820.434122672905, 353.6400165588405, 10944.061751765028])
    x = np.array([0.4, -1.3])
    print(GM.Error(data, y, x))
    
    norm_distr = np.zeros(r_x.shape[1])
    for i in range(r_x.shape[1]):
        norm_distr[i] = np.sqrt((data.dot(r_x[:n, i]) - y).dot(data.dot(r_x[:n, i]) - y))
    plt.plot(norm_distr)
    plt.show()
    
    
    
    return norm_distr