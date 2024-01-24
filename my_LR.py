import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

data = np.loadtxt("multiclass_data.csv", delimiter=',', dtype=float)
random.seed(16)
random.shuffle(data)
X = data[:, :-1]
Y = data[:, -1].reshape(data.shape[0], 1)

X_s = (X - np.mean(X)) / np.std(X)
X_s = np.column_stack((np.ones((X_s.shape[0], 1)), X_s))

X_train, X_test, Y_train, Y_test = train_test_split(X_s, Y, test_size=0.2, stratify=Y, random_state=1)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def gradient(X, Y, Theta):
    return np.dot(X.T, (sigmoid(X @ Theta) - Y))
def calculate_log_likelihood(Y, Y_hat):
    log_lh = np.mean(-Y * np.log(Y_hat + 1e-15) - (1 - Y) * np.log(1 - Y_hat + 1e-15))
    return log_lh
def create_mini_batches(X, Y, batch_size):
    mini_batches = []
    Y_pom = Y.copy()
    Y_pom = Y_pom.reshape((-1, 1))
    data = np.hstack((X, Y_pom))
    n_minibatches = np.shape(X)[0] // batch_size
    i = 0

    for i in range(n_minibatches):
        mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    if np.shape(X)[0] % batch_size != 0:
        mini_batch = data[(i + 1) * batch_size:np.shape(X)[0]]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))

    return mini_batches
def gradient_descent(X, Y, learning_rate, batch_size, max_iter):
    Theta_star = np.ones((X.shape[1], 1))
    Theta = np.ones((X.shape[1], 1))
    log_lh_arr = []
    m_mb_size = [0]
    minimum = float('inf')

    for _ in range(max_iter):
        mini_batches = create_mini_batches(X, Y, batch_size)
        i = 0
        for mini_batch in mini_batches:
            X_mini, Y_mini = mini_batch
            X_mini = np.array(X_mini)  # throws error if ignored
            Y_mini = np.array(Y_mini)  # throws error if ignored
            Theta -= learning_rate * gradient(X_mini, Y_mini, Theta)  # learning
            Y_hat = sigmoid(X_mini @ Theta)
            log_lh = calculate_log_likelihood(Y_mini, Y_hat)
            log_lh_arr.append(log_lh)
            m_mb_size.append(np.shape(X_mini)[0] + m_mb_size[len(m_mb_size) - 1])
            i += 1
            if (minimum > log_lh_arr[-1]):
                minimum = log_lh_arr[-1]
                Theta_star = Theta

    return Theta_star, log_lh_arr, m_mb_size[1:]
def calculate_multiclass_accuracy(actual_labels, predicted_labels):
    correct_count = 0
    total_count = len(actual_labels)
    for actual, predicted in zip(actual_labels, predicted_labels):
        if actual == predicted:
            correct_count += 1
    accuracy = 100 * correct_count / total_count
    return accuracy
def decision_making(X_train, X_test, Y_train, Y_test, alpha, batch_size, no_iter):
    Y_pom = Y_train.copy()
    indicator_0 = np.where(Y_train != 0)
    indicator_1 = np.where(Y_train == 0)
    Y_pom[indicator_0[0]] = 0
    Y_pom[indicator_1[0]] = 1
    Theta_star0, log_lh_arr0, m_mb_size = gradient_descent(X_train, Y_pom, alpha, batch_size, no_iter)

    Y_pom = Y_train.copy()
    indicator_0 = np.where(Y_train != 1)
    indicator_1 = np.where(Y_train == 1)
    Y_pom[indicator_0[0]] = 0
    Y_pom[indicator_1[0]] = 1
    Theta_star1, log_lh_arr1, m_mb_size = gradient_descent(X_train, Y_pom, alpha, batch_size, no_iter)

    Y_pom = Y_train.copy()
    indicator_0 = np.where(Y_train != 2)
    indicator_1 = np.where(Y_train == 2)
    Y_pom[indicator_0[0]] = 0
    Y_pom[indicator_1[0]] = 1
    Theta_star2, log_lh_arr2, m_mb_size = gradient_descent(X_train, Y_pom, alpha, batch_size, no_iter)

    Y_pred = np.empty(np.shape(X_test)[0])
    for i in range(np.shape(X_test)[0]):
        p1 = sigmoid(X_test[i, :] @ Theta_star0)
        p2 = sigmoid(X_test[i, :] @ Theta_star1)
        p3 = sigmoid(X_test[i, :] @ Theta_star2)
        list_p = [p1, p2, p3]
        Y_pred[i] = list_p.index(max(list_p))

    print("Accuracy[%]: ", calculate_multiclass_accuracy(Y_test, Y_pred))
    x_axis = np.array(m_mb_size)

    return x_axis, log_lh_arr0, log_lh_arr1, log_lh_arr2

# Optimal alpha, optimal batch_size
alpha_opt = 0.05
batch_size_opt = 64
no_iter = 20
x_axis, log_lh_arr0, log_lh_arr1, log_lh_arr2 = decision_making(X_train, X_test, Y_train, Y_test, alpha_opt, batch_size_opt, no_iter)
plt.figure()
plt.title('Optimal alpha, optimal minibatch size')
plt.plot(x_axis, log_lh_arr0, x_axis, log_lh_arr1, x_axis, log_lh_arr2)
plt.legend(['0', '1', '2'])
plt.show()

# Optimal alpha, big batch_size
batch_size_big = np.shape(X_train)[0]  # ja msm da je ovo np.shape(X)[0]
x_axis, log_lh_arr0, log_lh_arr1, log_lh_arr2 = decision_making(X_train, X_test, Y_train, Y_test, alpha_opt, batch_size_big, no_iter)
plt.figure()
plt.title('Optimal alpha, big minibatch size')
plt.plot(x_axis, log_lh_arr0, x_axis, log_lh_arr1, x_axis, log_lh_arr2)
plt.legend(['0', '1', '2'])
plt.show()

# Optimal alpha, small batch_size
batch_size_small = 8
x_axis, log_lh_arr0, log_lh_arr1, log_lh_arr2 = decision_making(X_train, X_test, Y_train, Y_test, alpha_opt, batch_size_small, no_iter)
plt.figure()
plt.title('Optimal alpha, small minibatch size')
plt.plot(x_axis, log_lh_arr0, x_axis, log_lh_arr1, x_axis, log_lh_arr2)
plt.legend(['0', '1', '2'])
plt.show()

# Big alpha, optimal batch_size
alpha_big = 1
x_axis, log_lh_arr0, log_lh_arr1, log_lh_arr2 = decision_making(X_train, X_test, Y_train, Y_test, alpha_big, batch_size_opt, no_iter)
plt.figure()
plt.title('Big alpha, optimal minibatch size')
plt.plot(x_axis, log_lh_arr0, x_axis, log_lh_arr1, x_axis, log_lh_arr2)
plt.legend(['0', '1', '2'])
plt.show()

# Small alpha, optimal batch_size
alpha_small = 0.005
x_axis, log_lh_arr0, log_lh_arr1, log_lh_arr2 = decision_making(X_train, X_test, Y_train, Y_test, alpha_small, batch_size_opt, no_iter)
plt.figure()
plt.title('Small alpha, optimal minibatch size')
plt.plot(x_axis, log_lh_arr0, x_axis, log_lh_arr1, x_axis, log_lh_arr2)
plt.legend(['0', '1', '2'])
plt.show()