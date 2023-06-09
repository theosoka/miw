import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions


class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


class MultiClassPerceptron:
    def __init__(self, eta=0.1, n_iter=100):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.perceptrons = {}

        for clazz in self.classes:
            y_class = np.where(y == clazz, 1, -1)
            ppn = Perceptron(self.eta, self.n_iter)
            ppn.fit(X, y_class)
            self.perceptrons[clazz] = ppn

        return self

    def predict(self, X):
        labels = []
        predictions = []
        for clazz, ppn in self.perceptrons.items():
            ppn_prediction = ppn.net_input(X)
            predictions.append(ppn_prediction)
            labels.append(clazz)
        predictions_id = np.argmax(np.array(predictions), axis=0)
        return np.array([labels[prediction_id] for prediction_id in predictions_id])

iris_ds = datasets.load_iris()
X = iris_ds.data[:, [2, 3]]
y = iris_ds.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
model = MultiClassPerceptron(eta=0.1, n_iter=1000)
model.fit(X_train, y_train)

plot_decision_regions(X=X_train, y=y_train, classifier=model)