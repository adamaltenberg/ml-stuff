import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Adaline:
    """ADAptive LInear NEuron classifier"""
    
    def __init__(self, learning_rate : float = 0.05, n_iter : int = 50, random_state: int = 1):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X: NDArray[np.float64], y: NDArray[np.int8]) -> Adaline:
        """
        """
        rng = np.random.RandomState(self.random_state)
        self.w_ = rng.normal(loc = 0.0, scale = 0.01, size = X.shape[1])
        self.b_ = np.float64(0.)
        self.losses_ = []

        for _ in range(self.n_iter):
            z = self.net_input(X)
            y_hat = self.activation(z)
            errors = y-y_hat
            self.w_ += self.learning_rate * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.learning_rate * 2.0 * errors.mean()
            self.losses_.append((errors**2).mean())
        return self
    
    def net_input(self, X: NDArray[np.float64]) -> np.float64:
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute linear activation"""
        return X
    
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.int8]:
        """return predicted class labels"""
        return np.where(self.activation(self.net_input(x)) >= 0.5, 1, 0)
    
    def plot_decision_regions(self, X: NDArray[np.float64], y: NDArray[np.int8], resolution: float = 0.02) -> None:
        '''plot the predicted decision boundaries/regions'''
        # marker generator and colour map
        names = ('Setosa', 'Versicolor', 'Virginica')
        markers = ('o', 's', '^', 'v', '<')
        colours = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colours[:len(np.unique(y))])

        # decision surface (contour plot)
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        lab = self.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        lab = lab.reshape(xx1.shape)
        plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        
        # plot class samples
        for i, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0],
                       y=X[y == cl, 1],
                       alpha=0.8,
                       color=colours[i],
                       marker=markers[i],
                       label=names[cl],
                       edgecolors='black')
        
        # plot each class
        for i, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0],
                        y=X[y == cl, 1],
                        alpha=0.8,
                        c=colours[i],
                        marker=markers[i],
                        label=f'Class {names[i]}',
                        edgecolor='black')
