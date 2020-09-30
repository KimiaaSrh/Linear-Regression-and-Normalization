def minibatch_weighted_gradient_descent(X,y,theta,learning_rate=0.01,iterations=10,batch_size =20):
    cov=np.cov(normal[:,[0,1]])
    for i in range(len(cov)):
        for j in range(len(cov)):
            if (i!=j):
                cov[i][j]=0
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,2))
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
            theta = theta -(1/m)*learning_rate*cov[i][i]*( X_i.T.dot((prediction - y_i)))
            cost += cal_cost(theta,X_i,y_i)
        theta_history[it,:] =theta.T
        cost_history[it]  = cost
    return theta, cost_history,theta_history
