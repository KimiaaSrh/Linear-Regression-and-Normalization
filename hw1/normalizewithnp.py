import numpy as np
from numpy import array, dot, transpose
from numpy.linalg import inv
import  pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def linear_regression(x_train, y_train):

    X = np.array(x_train)
    ones = np.ones(len(X))
    X = np.column_stack((ones,X))
    y = np.array(y_train)

    Xt = transpose(X)
    product = dot(Xt, X)
    theInverse = inv(product)
    w = dot(dot(theInverse, Xt), y)
    return w

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

def stocashtic_gradient_descent(X,y,theta,learning_rate=0.01,iterations=10):

    m = len(y)
    cost_history = np.zeros(iterations)

    temp=[]
    for it in range(iterations):
        cost =0.0
        temp.append(theta)
        for i in range(m):
            rand_ind = np.random.randint(0,m)
            X_i = X[rand_ind,:].reshape(1,X.shape[1])
            y_i = y[rand_ind].reshape(1,1)
            prediction = np.dot(X_i,theta)

            theta = theta -(1/m)*learning_rate*( X_i.T.dot((prediction - y_i)))
            cost += cal_cost(theta,X_i,y_i)
        cost_history[it]  = cost

    return theta, cost_history,temp

def minibatch_gradient_descent(X,y,theta,learning_rate=0.01,iterations=10,batch_size =20):

    m = len(y)
    cost_history = np.zeros(iterations)
    n_batches = int(m/batch_size)

    for it in range(iterations):
        cost =0.0
        indices = np.random.permutation(m)
        X = X[indices]
        y = y[indices]
        for i in range(0,m,batch_size):
            X_i = X[i:i+batch_size]
            y_i = y[i:i+batch_size]

            X_i = np.c_[np.ones(len(X_i)),X_i]

            prediction = np.dot(X_i,theta)

            theta = theta -(1/m)*learning_rate*( X_i.T.dot((prediction - y_i)))
            #thetas.append(theta)
            cost += cal_cost(theta,X_i,y_i)
        cost_history[it]  = cost

    return theta, cost_history


a = np.loadtxt('Data_Normal.txt', skiprows=1, delimiter=';')

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
#print (a)
#print (X_std)
#print (x_train)
#print (y_train)
b=linear_regression(x_train,y_train)
    # plotting regression line
#plot_regression_line(np.array(x_women),np.array(y_women),np.array(x_men),np.array(y_men),x_train, b)
###PART A STOCASTIC GRADIENT DESCENT ***********************************************

lr =0.1
n_iter = 200

theta = np.random.randn(2,1)

X_b = np.c_[np.ones((len(x_train),1)),x_train]
theta,cost_history,temp = stocashtic_gradient_descent(X_b,y_train,theta,lr,n_iter)
plot_regression_line(np.array(x_women),np.array(y_women),np.array(x_men),np.array(y_men),x_train, theta)

#print (temp)
#print(np.concatenate(temp,axis=1))
temp=np.array(temp)
#print (temp.reshape(-1,2))
#PART A Q5

theta0=[]
theta1=[]
theta0=temp[:,0]
theta1=temp[:,1]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(np.array(theta0), np.array(theta1), np.array(cost_history).reshape(-1,1),rstride=10, cstride=10)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


#plot_regression_line(np.array(x_women),np.array(y_women),np.array(x_men),np.array(y_men),x_train, theta)


###PART A  batch gradient descent (2000 epochs)  ***********************************************

#theta,cost_history = minibatch_gradient_descent(x_train,y_train,theta,lr,n_iter)
#plot_regression_line(np.array(x_women),np.array(y_women),np.array(x_men),np.array(y_men),x_train, theta)
