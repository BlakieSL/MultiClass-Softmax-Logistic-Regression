import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    y_train0 = y_train.copy()
    y_train1 = y_train.copy()
    y_train2 = y_train.copy()

    y_train0[y_train == 0] = 1
    y_train0[y_train != 0] = 0

    y_train1[y_train == 1] = 1
    y_train1[y_train != 1] = 0

    y_train2[y_train == 2] = 1
    y_train2[y_train != 2] = 0

    lrgd0 = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    lrgd0.fit(X_train, y_train0)

    lrgd1 = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    lrgd1.fit(X_train, y_train1)

    lrgd2 = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    lrgd2.fit(X_train, y_train2)

    scores0 = lrgd0.net_input(X_test)
    print(scores0)
    scores1 = lrgd1.net_input(X_test)
    scores2 = lrgd2.net_input(X_test)
    combined_scores = np.column_stack((scores0, scores1, scores2))

    exp_scores = np.exp(combined_scores - np.max(combined_scores, axis=1, keepdims=True))
    softmax_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    print(softmax_probs[:, 0])
    print(softmax_probs[:, 1])
    print(softmax_probs[:, 2])


if __name__ == '__main__':
    main()