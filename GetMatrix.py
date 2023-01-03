import numpy as np
import matplotlib.pyplot as plt
import math
import tqdm
import pandas  as pd

def ReadData(filename='insurance.csv'):
    df = pd.read_csv(filename)
    df.region = pd.factorize(df.region)[0]
    df.sex = pd.factorize(df.sex)[0]
    df.smoker = pd.factorize(df.smoker)[0]
    data = df.to_numpy()
    A = np.copy(data[:, :])
    b = np.copy(data[:, 6])
    A[:, 6] = 1
    #print(b)
    return A, b

def Error(A, b, x):
    return (A.dot(x)-b).dot(A.dot(x)-b)**(0.5) / A.shape[0]