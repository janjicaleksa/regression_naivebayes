import numpy as np
import matplotlib.pyplot as plt
import random

data = np.loadtxt("data.csv", delimiter=',', dtype=float)
X = data[:, :-1]
Y = data[:, -1].reshape(data.shape[0], 1)

def augment_features(X):
    m, n = X.shape
    X_squared = X**2
    X_cross = []
    for i in range(n-1):
        for j in range(i+1, n):
            X_cross.append((X[:, i]*X[:, j]).reshape((m, 1)))
    X_cross = np.column_stack(X_cross)
    return np.column_stack((X, X_squared, X_cross))

X = augment_features(X)

def my_train_test_split(X, Y, test_size=0.2, random_seed=16):
    np.random.seed(random_seed)
    num_samples = len(X)
    num_test_samples = int(test_size * num_samples)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]
    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = my_train_test_split(X, Y, test_size=0.2, random_seed=16)

def my_standardization(X_train, Y_train, X, Y):
    X_mean, X_std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    X = (X-X_mean)/X_std
    Y_mean = np.mean(Y_train)
    Y = Y-Y_mean
    return X, Y

X_test, Y_test = my_standardization(X_train, Y_train, X_test, Y_test)
X_train, Y_train = my_standardization(X_train, Y_train, X_train, Y_train)

def LASSO_train(X, Y, Lambda, max_iterations=1000, learning_rate=0.01):
    Theta = np.ones((X.shape[1], 1))
    for _ in range(max_iterations):
        gradient = X.T@(X@Theta - Y)/Y.shape[0] + Lambda*np.sign(Theta)
        Theta -= learning_rate*gradient
    return Theta

def my_k_fold_cross_validation(X, Y, num_folds, Lambdas):
    m, n = X.shape
    indices = np.arange(m)
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    folds_X = np.array_split(X, num_folds)
    folds_Y = np.array_split(Y, num_folds)
    RMSEs = []
    for Lambda in Lambdas:
        RMSE = 0
        for i in range(num_folds):
            X_val, Y_val = folds_X[i], folds_Y[i]
            train_indices = [j for j in range(num_folds) if j != i]
            X_train = np.concatenate([folds_X[j] for j in train_indices], axis=0)
            Y_train = np.concatenate([folds_Y[j] for j in train_indices], axis=0)
            Theta = LASSO_train(X_train, Y_train, Lambda)
            RMSE += np.sqrt(np.mean((X_val@Theta - Y_val)**2))
        RMSEs.append(RMSE/num_folds)
    plt.plot(Lambdas, RMSEs)
    plt.ylabel("RMSE")
    plt.xlabel("$\lambda$")
    return Lambdas[np.argmin(RMSEs)]

best_Lambda = my_k_fold_cross_validation(X_train, Y_train, num_folds=5, Lambdas=np.linspace(0.1,10,200))
Theta = LASSO_train(X_train, Y_train, best_Lambda)

print("Optimal regularization parameter lambda: ", best_Lambda)
RMSE_train = np.sqrt(np.mean((X_train@Theta-Y_train)**2))
print("Root Mean Squared Error - Train Set: " + str(RMSE_train))
RMSE_test = np.sqrt(np.mean((X_test@Theta-Y_test)**2))
print("Root Mean Squared Error - Test Set: " + str(RMSE_test))

stats = np.array([np.mean(X_train,axis=0), np.std(X_train,axis=0), np.mean(Y_train)], dtype=object)
np.save('stats.npy', stats)
np.save('coefficients.npy', Theta)