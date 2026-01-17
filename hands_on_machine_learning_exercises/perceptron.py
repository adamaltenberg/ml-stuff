"""binary classification using the perceptron learning algorithm
"""

import numpy as np
from numpy.typing import NDArray

class Perceptron:
    """simple perceptron classifier for linearly seperable binary classification.
    
    weight_update = learning_rate * error * input.
    
    parameters
    ----------
    learning_rate : float, default=0.05
        controls the magnitude of adjustments made during training 
        must be between 0 and 1
    n_iter : int, default=50
        # of passes over the training dataset (epochs)
    random_state : int, default=1
        seed for the random number generator
        
    attributes
    ----------
    w_ : NDArray[np.float64]
        weights learned during training
        shape (n_features,)
    b_ : np.float64
        bias term learned during training
    errors_ : list of int
        # of misclassifications in each epoch
    """
    def __init__(self, learning_rate : float = 0.05, n_iter : int = 50, random_state: int = 1):
        """
        parameters
        ----------
        learning_rate : float, default=0.05
            between 0 and 1
        n_iter : int, default=50
            # of epochs
        random_state : int, default=1
            random seed for reproducibility
        """
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X: NDArray[np.float64], y: NDArray[np.int_]) -> Perceptron:
        """train the perceptron on the provided data
        weights are initialized from a normal distribution of very small values
        
        parameters
        ----------
        X : NDArray[np.float64], shape (n_samples, n_features)
            training feature matrix
            each row is a sample, each column is a feature
        y : NDArray[np.int_], shape (n_samples,)
            binary target labels
            
        returns
        -------
        self : Perceptron
            the fitted model
            
        attributes set
        ---------------
        w_ : NDArray[np.float64]
            weight vector of shape (n_features,).
        b_ : np.float64
            bias term
        errors_ : list of int
            # of misclassifications per epoch
        """
        rng = np.random.RandomState(self.random_state)
        
        self.w_ = rng.normal(loc = 0.0, scale = 0.01, size = X.shape[1])
        self.b_ = np.float64(0.)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                delta = self.learning_rate*(target - self.predict(xi))
                self.w_ += delta*xi
                self.b_ += delta
                errors += 1 if delta != 0.0 else 0
            self.errors_.append(errors)
        return self
    
    def net_input(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """calculate z (weighted sum) for input samples
        
        z = wTx + b
        
        parameters
        ----------
        x : NDArray[np.float64], shape (n_features,) or (n_samples, n_features)
            input samples
            
        returns
        -------
        NDArray[np.float64]
            z values
            same shape as input x
        """
        return np.dot(x, self.w_) + self.b_
    
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.int_]:
        """1 if z >= 0, else 0
        
        parameters
        ----------
        x : NDArray[np.float64], shape (n_features,) or (n_samples, n_features)
            input to be classified
            
        returns
        -------
        NDArray[np.int_]
            predicted class of input
        """
        return np.where(self.net_input(x) >= 0.0, 1, 0)