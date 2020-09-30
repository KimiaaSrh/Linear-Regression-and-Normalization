from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math

asli=[[2,10],[4,12]]
def wls(x,y):
    meanX=np.average(x)
    meanY=np.average(y)
    y=[meanX,meanY]
    distance=0;
    w=h=len(x)
    weights=[[0 for x in range(w)] for y in range(h)]
    for i in range (0,len(x)):
        distance = sum([(a - b) ** 2 for a, b in zip(asli[i], y)])
        weights[i][i]=distance
    return weights,meanX,meanY
print(wls([2,4],[10,12]))
