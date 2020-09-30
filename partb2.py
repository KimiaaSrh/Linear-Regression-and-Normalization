import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import ceil
from scipy import linalg
from IPython.display import Image
from IPython.display import display
plt.style.use('seaborn-white')

from math import ceil
import numpy as np
from scipy import linalg

#Defining the bell shaped kernel function - used for plotting later on
def kernel_function(xi,x0,tau= .005):
    return np.exp( - (xi - x0)**2/(2*tau)   )

def lowess_bell_shape_kern(x, y, tau = .005):
    """lowess_bell_shape_kern(x, y, tau = .005) -> yest
    Locally weighted regression: fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The kernel function is the bell shaped function with parameter tau. Larger tau will result in a
    smoother curve.
    """
    m = len(x)
    yest = np.zeros(60)

    #Initializing all weights from the bell shape kernel function
    w = np.array([np.exp(- (x - x[i])**2/(2*tau)) for i in range(m)])

    #Looping through all x-points
    for i in range(60):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        theta = linalg.solve(A, b)
        yest[i] = theta[0] + theta[1] * x[i]

    return yest

def plot_regression_line(x_women,y_women,x_men,y_men,x, b):
    # plotting the actual points as scatter plot

    plt.scatter(x_women, y_women, color = "r",
               marker = "*", s = 20)
    plt.scatter(x_men, y_men, color = "b",
               marker = "*", s = 20)

    # predicted response vector
    y_pred = b[0] + b[1]*x

    # plotting the regression line
    plt.plot(x, y_pred, color = "g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()

a = np.loadtxt('Data_With_Outlier.txt', skiprows=1, delimiter=';')

X_std = (a - a.min(axis=0)) / (a.max(axis=0) - a.min(axis=0))

x_women=[]
y_women=[]
x_men=[]
y_men=[]
sag=X_std[:,[2]]
x_train=X_std[:,[0]]
y_train=X_std[:,[1]]
for i in range(0,len(sag)) :
    if(sag[i]==0):
        x_women.append(x_train[i])
        y_women.append(y_train[i])
    else:
        x_men.append(x_train[i])
        y_men.append(y_train[i])
f = 0.25
thetaout = lowess_bell_shape_kern(x_train,y_train)
plot_regression_line(x_women,y_women,x_men,y_men,x_train, thetaout)
