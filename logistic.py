import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(0)


def gen_target(X):
    return (3.5 * X) > 5


n_records = 300
X = np.sort(np.random.rand(n_records))
y = gen_target(X) + np.random.randn(n_records) * 0.1
X = X.reshape(-1, 1)

print("Number of training examples : ", X.shape[0])
print("Number of predictors : ", y.shape[1] if len(y.shape) > 1 else 1)

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lr = LogisticRegression(
    fit_intercept=False, penalty="l1", solver="liblinear", random_state=0
)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("Coefficients : \n")
for ii, coef in enumerate(lr.coef_):
    plt.figure()
    plt.bar(range(len(coef)), coef)
    plt.xticks(range(len(coef)))
    plt.xlabel("Index")
    plt.ylabel("Coefficient")
    for jj, c in enumerate(coef):
        print("Class-{0:2d} : Coef-{1:2d} : {2:2.4f}".format(ii, jj, c))
    plt.show()

print("\nAccuracy Score: {:2.4f}".format(accuracy_score(y_test, y_pred)))

if X.shape[1] < 2:
    plt.figure()
    plt.scatter(X_test, y_test, color="black")
    plt.plot(X_test, y_pred, color="blue", linewidth=3)
    plt.show()
