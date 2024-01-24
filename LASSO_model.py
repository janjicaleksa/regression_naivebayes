import numpy as np

def augment_features(X):
    m, n = X.shape
    X_squared = X**2
    X_cross = []
    for i in range(n-1):
        for j in range(i+1, n):
            X_cross.append((X[:, i]*X[:, j]).reshape((m, 1)))
    X_cross = np.column_stack(X_cross)
    return np.column_stack((X, X_squared, X_cross))

def LASSO_model(X):
    X = augment_features(X)

    X_mean, X_std, Theta0 = np.load('stats.npy', allow_pickle=True)
    Theta = np.load('coefficients.npy')

    X = (X-X_mean)/X_std
    Y_pred = X@Theta + Theta0

    return Y_pred