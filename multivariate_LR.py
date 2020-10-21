# A simple linear regression model.
# we write one instead of using sklearn, to get the time-series loss and weights,
# for time-series visualization and interaction of the training process.

import numpy as np

class LinearRegression:
    def __init__(self, lam):
        """
        Class constructor.
        
        args:
            lam (float) : the regularizer value
        """
        self.lam = lam
        self.b = None
        self.theta = None
        return
        
    def loss(self, h, y):
        """
        Compute the average loss L(theta, b) based on the provided formula.
        
        args:
            h (np.array[n_samples]) : a vector of hypothesis function values on every input data point,
                this is the output of self.predict(X)
            y (np.array[n_samples]) : the output data vector
            
        return:
            float : the average loss value
        """
        
        n_samples = h.shape[0]
        
        error_sum = np.square(h - y).sum()
        
        regular_sum = np.square(self.theta).sum() * self.lam
        
        return (error_sum + regular_sum) / (2 * n_samples)
    
    def fit(self, X, y, n_iters = 100, alpha = 0.01):
        """
        Train the model weights and intercept term using batch gradient descent.
        
        args:
            X (np.array[n_samples, n_dimensions]) : the input data matrix
            y (np.array[n_samples]) : the output data vector

        kwargs:
            n_iters (int) : the number of iterations to train for
            alpha (float) : the learning rate
            
        return:
            List[float] : a list of length (n_iters + 1) that contains the loss value
                before training and after each training iteration
            List[np.array[n_dimensions + 1]] : containing the linear weights on each epoch.
        """
        
        # initial params and loss
        loss_values = []
        weights_on_epochs = []
        n_samples, n_dimensions = X.shape
        self.theta = np.zeros(n_dimensions)
        self.b = 0.0
        
        h = self.predict(X)
        loss_values.append(self.loss(h, y))
        weights_on_epochs.append(self.get_params())
        
        # training
        for _ in range(n_iters):
            diff = h - y
            
            # grad for b
            b_grad = diff.sum() * alpha / n_samples
            
            # grad for theta
            theta_grad = (diff.reshape(-1, 1) * X).sum(axis=0)
            theta_grad = theta_grad + self.lam * self.theta
            theta_grad = theta_grad * alpha / n_samples
            
            # update and new loss
            self.b = self.b - b_grad
            self.theta = self.theta - theta_grad
            
            h = self.predict(X)
            loss_values.append(self.loss(h, y))
            weights_on_epochs.append(self.get_params())
            
        
        return loss_values, weights_on_epochs
        
    def get_params(self):
        """
        Get the model weights and intercept term.
        
        return:
            Tuple(theta, b):
                theta (np.array[n_dimensions]) : the weight vector
                b (float) : the intercept term
        """
        return np.concatenate([self.theta, [self.b]])
    
    def predict(self, X):
        """
        Predict the output vector given an input matrix
        
        args:
            X (np.array[n_samples, n_dimensions]) : the input data matrix
        
        return:
            np.array[n_samples] : a vector of hypothesis function values on every input data point
        """
        return X @ self.theta + self.b