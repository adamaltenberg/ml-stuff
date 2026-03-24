import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_decision_regions(
    classifier,
    X: NDArray[np.float64],
    y: NDArray[np.int8],
    resolution: float = 0.02,
    mark_split: bool = False,
    test_idx: NDArray[np.int_] | list[int] | None = None,
) -> None:
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
    mark_split : bool, default=False
        if True, visually marks examples provided by `test_idx` as test samples
    test_idx : array-like of int, optional
        indices of test-set samples in X; used only when mark_split=True
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

    if mark_split and test_idx is not None:
        test_idx = np.asarray(test_idx)
        plt.scatter(
            X[test_idx, 0],
            X[test_idx, 1],
            c='none',
            edgecolors='black',
            alpha=1.0,
            linewidth=1,
            marker='o',
            s=100,
            label='Test set',
        )

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
    
if __name__ == "__main__":
    z = np.arange(-7, 7, 0.1)
    sigma_z = sigmoid(z)
    plt.plot(z, sigma_z)
    plt.axvline(0.0, color='k')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('z')
    plt.ylabel('$\sigma (z)$')
    # y axis ticks and gridline
    plt.yticks([0.0, 0.5, 1.0])
    ax = plt.gca()
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.show()

    def loss_1(z):
        return - np.log(sigmoid(z))
    def loss_0(z):
        return - np.log(1 - sigmoid(z))
    z = np.arange(-10, 10, 0.1)
    sigma_z = sigmoid(z)
    c1 = [loss_1(x) for x in z]
    plt.plot(sigma_z, c1, label='L(w, b) if y=1')
    c0 = [loss_0(x) for x in z]
    plt.plot(sigma_z, c0, linestyle='--', label='L(w, b) if y=0')
    plt.ylim(0.0, 5.1)
    plt.xlim([0, 1])
    plt.xlabel('$\sigma(z)$')
    plt.ylabel('L(w, b)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()