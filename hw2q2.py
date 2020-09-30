import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
import math

mean1 = [2, 4]
covariance1 = [ [1, 0], [0, 1] ]

class1 = [[],[]]
class1[0], class1[1] = np.random.multivariate_normal(mean1, covariance1, 100).T


mean2 = [1, 7]
covariance2 = [ [1, 0], [0, 1] ]

class2 = [[], []]
class2[0], class2[1] = np.random.multivariate_normal(mean2, covariance2, 140).T

dataset = [[], []]
for i in range(100):
    dataset[0].append(class1[0][i])
    dataset[1].append(class1[1][i])
for i in range(140):
    dataset[0].append(class2[0][i])
    dataset[1].append(class2[1][i])

Y = []
for i in range(100):
    Y.append(0)
for i in range(140):
    Y.append(1)

def draw_samples(dataset, Y):
    plt.scatter(dataset[0], dataset[1], c = Y)
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.show()

draw_samples(dataset, Y)


def likelihood_X_yi(X, i):
    if(i == 0):
        mue = mean1
    else:
        mue = mean2
    P = math.exp( -0.5 * ( pow(X[0] - mue[0], 2) + pow(X[1] - mue[1], 2) ) ) / pow( math.sqrt(2 * math.pi), 2 )
    return P;

def prior(i):
    if(i == 0):
        return 100 / 240
    else:
        return 140 / 240

def classify(dataset, Y):
    c1 = [[], []]
    c2 = [[], []]
    mistaken = [[], []]
    for i in range(240):
        Xi = [dataset[0][i], dataset[1][i]]
        P0 = likelihood_X_yi(Xi, 0) * prior(0)
        P1 = likelihood_X_yi(Xi, 1) * prior(1)

        if(P0 > P1 and Y[i] == 0):
            c1[0].append(Xi[0])
            c1[1].append(Xi[1])

        elif(P1 > P0 and Y[i] == 1):
            c2[0].append(Xi[0])
            c2[1].append(Xi[1])
        else:
            mistaken[0].append(Xi[0])
            mistaken[1].append(Xi[1])

    plt.scatter(c1[0], c1[1], marker = ".")
    plt.scatter(c2[0], c2[1], marker = "+")
    plt.scatter(mistaken[0], mistaken[1], marker = "*")
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.show()

classify(dataset, Y)
