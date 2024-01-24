import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split

data = pd.read_csv("multiclass_data.csv", delimiter=',', names=['x1', 'x2', 'x3', 'x4', 'x5', 'y'])
X = data.drop('y', axis=1)
Y = data['y']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)
def data_stats(X_train, Y_train):
    Mi = np.zeros((len(Y_train.value_counts()), X_train.shape[1]))
    Sigma = np.zeros((len(Y_train.value_counts()), X_train.shape[1]))
    for cat in range(len(Y_train.value_counts())):
        index_list = Y_train[Y_train == cat].index
        X = X_train.loc[index_list]
        Mi[cat, :] = X.mean().to_numpy()
        Sigma[cat, :] = X.std().to_numpy()
    return Mi, Sigma
def gauss(X, Y, Mi, Sigma):
    return (1/(np.sqrt(2*math.pi*Sigma[Y, :]**2)))*np.exp(-((X-Mi[Y, :])**2)/(2*Sigma[Y, :]**2))
def decision_making(Mi, Sigma, X_test, Y_test, Phi):
    P = np.zeros((len(Y_test), len(Y_test.value_counts())))
    i = 0
    for _, row in X_test.iterrows():
        for cat in range(len(Y_test.value_counts())):
            num = Phi[cat]*math.prod(gauss(row, cat, Mi, Sigma))
            denum = sum([Phi[j]*math.prod(gauss(row, j, Mi, Sigma)) for j in range(len(Y_test.value_counts()))])
            P[i, cat] = num/denum
        i += 1
    return P

Mi, Sigma = data_stats(X_train, Y_train)
Phi = Y_train.value_counts()/Y_train.count()

P_train = decision_making(Mi, Sigma, X_train, Y_train, Phi)
Y_pred_train = [np.argmax(P_train[i]) for i in range(len(P_train))]
print(f"Accuracy[%] - training set: {100*sum(Y_pred_train == Y_train)/len(Y_train)}")

P_test = decision_making(Mi, Sigma, X_test, Y_test, Phi)
Y_pred_test = [np.argmax(P_test[i]) for i in range(len(P_test))]
print(f"Accuracy[%] - test set: {100*sum(Y_pred_test == Y_test)/len(Y_test)}")