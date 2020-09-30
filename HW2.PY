from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB


simplefilter(action='ignore', category=FutureWarning)

def main_logistic_reg () :
    iris = load_iris()
    X  = iris.data
    X = np.c_[X[50: ,2],X[50:,3]]
    Y = []
    for i in range(len(X)):
        if i < 50 :
            Y.append(1) #virginica
        else:
            Y.append(0)#viricolor

    Y=np.array(Y)
    plt.scatter(X[:,0],X[:,1],c=Y)
    plt.show()

def show_logistic_regression():
    iris =load_iris()
    X = 2
    Y = 3
    formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

    plt.figure(figsize=(8, 8))
    plt.scatter(iris.data[50:, X], iris.data[50:, Y], c=iris.target[50:150])
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    plt.xlabel(iris.feature_names[X])
    plt.ylabel(iris.feature_names[Y])

    plt.tight_layout()
    plt.show()

def decision_boundary():
    iris = load_iris()
    X =iris.data[50: ,2:]
    Y =iris.target[50:]

    df = pd.DataFrame(X,columns = ['petal length','petal width'])
    df.head()
    #print(df)

    #for training
    X=iris.data
    X=np.c_[X[50: ,2],X[50:,3]]
    Y= []
    for i in range(len(X)):
        if i <50 :
            Y.append(1)
        else :
            Y.append(0)
    Y = np.array(Y)
    #__________________________________
    prem = np.random.permutation(len(X))
    x_train ,x_test = x[prem][20:],x[prem][:20]
    y_train , y_test = y[prem][20:],y[prem][:20]

    print(x_train.shape , y_train.shape , x_test.shape , y_test.shape)
    log = LogisticRegression()
    log.fit(x,y)

    w,b =log.coef_ , log.intercept_
    probas = log.predict_proba(x_train)
    plt.scatter(X[:,0],X[:,1],c=Y)
    ax = plt.gca()
    ax.autoscale =False
    xvals = np.array(ax.get_xlim())
    yvals = -(xvals * w[0][0]+b)/w[0][1]
    plt.plot(xvals,yvals)
    #plt.show() #decision_boundary for training data

    predict = log.predict(x_test)
    print(predict)
    #print(len(x_test) , len(y_test))
    plt.scatter(x_test[:,0],x_test[:,1],c=y_test)
    ax =plt.gca()
    xvals =np.array(ax.get_xlim())
    yvals = -(xvals*w[0][0]+b)/w[0][1]
    plt.plot(xvals,yvals)
    plt.show()

show_logistic_regression()
main_logistic_reg ()
decision_boundary()

new_Dataset= iris.data[50:, 2:]


print("new data set : \n",new_Dataset)
print(type(new_Dataset))

show_plot()


iris = load_iris()
X= iris.data[50: ,2:]
y=iris.target[50:]
