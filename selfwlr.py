def linear_regression(x_train, y_train):

        m = len(x_train)

        w = np.array([np.exp(- (x_train - x_train[i])**2/2) for i in range(m)])

        for i in range(m):
            x_train[i] = 



    X = np.array(x_train)
    ones = np.ones(len(X))
    X = np.column_stack((ones,X))
    y = np.array(y_train)

    Xt = transpose(X)
    product = dot(Xt, X)
    theInverse = inv(product)
    w = dot(dot(theInverse, Xt), y)
    return w
