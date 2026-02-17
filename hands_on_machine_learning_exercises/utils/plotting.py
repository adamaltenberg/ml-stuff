import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_decision_regions(classifier, X: NDArray[np.float64], y: NDArray[np.int8], resolution: float = 0.02) -> None:
    """Plot the predicted decision boundaries/regions for a fitted classifier.
    
    parameters
    ----------
    classifier : object
        a fitted classifier with a predict method
    X : NDArray[np.float64], shape (n_samples, 2)
        feature matrix (must be 2D for plotting)
    y : NDArray[np.int8], shape (n_samples,)
        target labels
    resolution : float, default=0.02
        step size for the meshgrid
    """
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
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
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
