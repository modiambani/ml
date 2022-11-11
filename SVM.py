import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(0)
from sklearn import datasets

X1, y1 = datasets.load_iris(return_X_y=True)
X1 = X1[:, 3].reshape(-1, 1)


def gen_target(X):
    return (X) > 0.5


n_records = 300
X2 = np.sort(np.random.rand(n_records))
y2 = gen_target(X2)
X2 = X2.reshape(-1, 1)
print("Number of training examples : ", X1.shape[0])
print("Number of predictors : ", y1.shape[1] if len(y1.shape) > 1 else 1)

print("Number of training examples : ", X2.shape[0])
print("Number of predictors : ", y2.shape[1] if len(y2.shape) > 1 else 1)


def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    sv = SVC(kernel="rbf", gamma=5, probability=True)
    sv.fit(X_train, y_train)
    y_pred = sv.predict(X_test)
    print("\nAccuracy Score: {:2.4f}".format(accuracy_score(y_test, y_pred)))
    print(f"No Of Support Vectors : {sv.n_support_}")

    if X.shape[1] < 2:
        plt.figure()
        plt.scatter(X_train, y_train, color="red")
        plt.scatter(X_test, y_pred, color="blue", linewidth=3)
        plt.show()


train(X1, y1)
train(X2, y2)
