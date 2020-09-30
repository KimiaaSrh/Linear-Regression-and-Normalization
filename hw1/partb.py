import numpy as np
from numpy import array, dot, transpose
from numpy.linalg import inv
import  pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

a = np.loadtxt('Data_Normal.txt', skiprows=1, delimiter=';')

X_std = (a - a.min(axis=0)) / (a.max(axis=0) - a.min(axis=0))

x_women=[]
y_women=[]
x_men=[]
y_men=[]
jensiat=X_std[:,[2]]
x_train=X_std[:,[0]]
y_train=X_std[:,[1]]

for i in range(0,len(jensiat)) :
    if(jensiat[i]==0):
        x_women.append(x_train[i])
        y_women.append(y_train[i])
    else:
        x_men.append(x_train[i])
        y_men.append(y_train[i])

x_train = np.array(x_train)
ones = np.ones(len(x_train))
x_train = np.column_stack((ones,x_train))

asli=X_std[:,[0,1]]

def  cal_cost(theta,X,y):
    '''

    Calculates the cost for given X and Y. The following shows and example of a single dimensional X
    theta = Vector of thetas
    X     = Row of X's np.zeros((2,j))
    y     = Actual y's np.zeros((2,1))

    where:
        j is the no of features
    '''

    m = len(y)

    predictions = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions-y))
    return cost

def weightedlistsquare(x,y):
    meanX=np.average(x)
    meanY=np.average(y)
    y=[meanX,meanY]
    distance=0;
    w=h=len(x)
    weights=[[0 for x in range(w)] for y in range(h)]
    for i in range (0,len(x)):
        distance = sum([(a - b) ** 2 for a, b in zip(asli[i], y)])
        weights[i][i]=1/distance

    inverse=inv(dot(dot(transpose(x_train),weights ),x_train))
    second=dot(dot(transpose(x_train),weights),y_train)
    result = dot(inverse,second)
    #print (np.array(result))
    return result

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

def batch_gradient_descent(x_t,y,theta,learning_rate=0.01,iterations=100):

    meanX=np.average(x_t)
    meanY=np.average(y)
    y=[meanX,meanY]
    distance=0;
    w=h=len(x_t)
    weights=[[0 for x in range(w)] for y in range(h)]
    for i in range (0,len(x_t)):
        distance = sum([(a - b) ** 2 for a, b in zip(asli[i], y)])
        weights[i][i]=1/distance

    m = len(y)
    cost_history = np.zeros(iterations)

    temp=[]
    for it in range(iterations):
        cost =0.0
        temp.append(theta)
        for i in range(m):
            rand_ind = np.random.randint(0,m)
            X_i = x_t[rand_ind,:].reshape(1,x_t.shape[1])
            y_i = y[rand_ind].reshape(1,1)
            prediction = np.dot(X_i,theta)
            theta = theta -(1/m)*learning_rate*weights[i][i]*( X_i.T.dot((prediction - y_i)))
            cost += cal_cost(theta,X_i,y_i)
        cost_history[it]  = cost

    return theta, cost_history,temp



b=weightedlistsquare(x_train,y_train)
#print (np.array(b))
    # plotting regression line
x_train=X_std[:,[0]]
#plot_regression_line(np.array(x_women),np.array(y_women),np.array(x_men),np.array(y_men),x_train, np.array(b))

lr =0.1
n_iter = 200

theta = np.random.randn(2,1)
X_b = np.c_[np.ones((len(x_train),1)),x_train]
theta,cost_history,temp = batch_gradient_descent(X_b,y_train,theta,lr,n_iter)
plot_regression_line(np.array(x_women),np.array(y_women),np.array(x_men),np.array(y_men),x_train, np.array(theta))
