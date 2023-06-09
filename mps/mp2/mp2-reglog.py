import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions


class MultiRegression:

    def __init__(self, eta=0.05, n_iter=1000, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(x)
            output = self.activation(net_input)  # apply the sigmoid function to the weighted sum to obtain a
            # probability estimate
            errors = (y - output)
            self.w_[1:] += self.eta * x.T.dot(errors)  # update weights for input features using gradient descent
            self.w_[0] += self.eta * errors.sum()
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))  # calculate the cost function for the
            # current iteration
            self.cost_.append(cost)

        return self

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, 0)

    def probability(self, X):
        net_input = self.net_input(X)
        return self.activation(net_input)  # obtain the estimated probability for a sample

    def train_reglogs(self, X, y):
        reglogs = []  # a list to hold the binary classifiers for each class
        for class_value in np.unique(y):
            train_data_classes = y.copy()
            train_data_classes[(y == class_value)] = 1  # current class to 1
            train_data_classes[(y != class_value)] = 0  # other classes to 0
            rl = MultiRegression(eta=self.eta, n_iter=self.n_iter,
                                 random_state=self.random_state)
            rl.fit(X, train_data_classes)
            rl.class_value = class_value
            reglogs.append(rl)
        return reglogs

    def classify(self, reglogs, test_data):
        predictions = []
        for sample in test_data:
            activated_classes = []
            for reglog in reglogs:
                if np.argmax(reglog.predict(sample)) == 1:
                    activated_classes.append(reglog.class_value)
            predictions.append(activated_classes)
        return predictions


def test_sample(sample, reglog):
    return reglog.predict(sample) == 1


def data():
    iris = datasets.load_iris()
    x = iris.data[:, [2, 3]]
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=1, stratify=y)
    return x_train, x_test, y_train, y_test


def print_probabilities(data, classes, reglogs, predictions):
    for sample, classes_value, predicted_value in zip(data, classes, predictions):
        sample_probabilites = {}
        probabilities_sum = 0
        for reglog in reglogs:
            p = reglog.probability(sample)
            probabilities_sum += p
            p = round(p, 2)
            sample_probabilites[reglog.class_value] = p
        print(
            f'Sample {sample}, real value: {classes_value}, predicted: {predicted_value}, probability: {round(probabilities_sum, 2)}')


def main():
    train_data, test_data, train_data_classes, test_data_classes = data()
    mlr = MultiRegression()
    mlr.fit(train_data, train_data_classes)
    reglogs = mlr.train_reglogs(train_data, train_data_classes)
    predictions = mlr.classify(reglogs, test_data)
    plot_decision_regions(X=train_data, y=train_data_classes, classifier=mlr)
    print_probabilities(test_data, test_data_classes, reglogs, predictions)


if __name__ == '__main__':
    main()
