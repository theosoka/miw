from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl,
                    edgecolor='black')


iris = datasets.load_iris()
X = iris.data[:, [1, 2]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)

tree_entropy = tree.DecisionTreeClassifier(criterion='entropy')
tree_entropy.fit(X_train, y_train)

plot_decision_regions(X_test, y_test, classifier=tree_entropy)
plt.title('Decision Tree - Entropy')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='upper left')
plt.show()

tree_gini = tree.DecisionTreeClassifier(criterion='gini')
tree_gini.fit(X_train, y_train)

plot_decision_regions(X_test, y_test, classifier=tree_gini)
plt.title('Decision Tree - Gini')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='upper left')
plt.show()

for depth in [3, 6, 9]:
    tree_gini = tree.DecisionTreeClassifier(criterion='gini', max_depth=depth)
    tree_gini.fit(X_train, y_train)

    plot_decision_regions(X_test, y_test, classifier=tree_gini)
    plt.title(f'Decision Tree - Gini (max_depth={depth})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper left')
    plt.show()

forest_entropy_small = RandomForestClassifier(criterion='entropy', n_estimators=10)
forest_entropy_small.fit(X_test, y_test)

plot_decision_regions(X_test, y_test, classifier=forest_entropy_small, test_idx=range(105, 150))
plt.title('Random Forest - Entropy (n_estimators=10)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='upper left')
plt.show()

