#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 
# Created by liutongtong on 2019-03-25 21:51
#

"""This module implements perceptron with Python3.

The algorithms are based on Chapter 2 of the book 统计学习方法 (李航, 2012).
"""

import numpy as np


class Perceptron(object):
    """Implementation of perceptron.

    Attributes:
        coef_ (array, shape (n_features,)): Weights assigned to the features.
        intercept_ (double): Bias in decision function.
        n_iter_ (int): The actual number of iterations to reach the stopping criterion.

        fit: Fit perceptron with Stochastic Gradient Descent.
        predict: Predict class labels for samples in X.
        score: Returns the mean accuracy on the given test data and labels.
    """

    def __init__(self, max_iter=1000, tol=0.0, shuffle=True, eta0=1.0, random_state=None):
        """Inits perceptron.

        Args:
            max_iter (int): The maximum number of passes over the training data (aka epochs). Defaults to 1000.
            tol (float): The criterion to check whether a sample is classified correctly.
                A sample is classified correctly if y*(wX+b) > -tol. tol should be non-negative.
                Defaults to 0.
            shuffle (bool): Whether or not the training data should be shuffled after each epoch. Defaults to True.
            eta0 (float): Constant by which the updates are multiplied. Defaults to 1.
            random_state (int): The seed of the pseudo random number generator to use when shuffling the data.
        """
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0

        self.max_iter = max_iter
        self.tol = max(tol, 0.0)
        self.shuffle = shuffle
        self.eta0 = eta0
        self.random_state = random_state
        np.random.seed(self.random_state)

    def fit(self, X, y):
        """Fit linear model with Stochastic Gradient Descent.

        Args:
            X (array-like, shape (n_samples, n_features)): Training data.
            y (array, shape (n_samples,)): Target values.

        Returns:
            self: Returns an instance of self.

        Raise:
            ValueError: An error occurred when n_samples of X and y are inconsistent.
        """
        if len(X) != len(y):
            raise ValueError(f"Found input variables with inconsistent numbers of samples: {len(X)}, {len(y)}")

        X, y = np.array(X), np.array(y)
        self.coef_ = np.ones(len(X[0]), dtype=np.float32)
        self.intercept_ = 0.0
        for self.n_iter_ in range(1, self.max_iter + 1):
            if self.shuffle:
                shuffled_indices = np.arange(len(X))
                np.random.shuffle(shuffled_indices)
                X = X[shuffled_indices]
                y = y[shuffled_indices]
            update = False
            for i in range(len(X)):
                if y[i] * (np.dot(self.coef_, X[i]) + self.intercept_) <= -self.tol:
                    self.coef_ += self.eta0 * y[i] * X[i]
                    self.intercept_ += self.eta0 * y[i]
                    update = True
            if not update:
                break
        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X (array-like, shape (n_query, n_features)): Samples.

        Returns:
            y (array, shape (n_query,)): Predicted class label per query.
        """
        return np.sign(np.dot(self.coef_, np.array(X).T) + self.intercept_)

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X (array-like, shape (n_query, n_features)): Test samples.
            y (array, shape (n_query,)): True labels for X

        Returns:
            score (float): Mean accuracy of self.predict(X) wrt. y.
        """
        return (self.predict(X) == np.array(y)).sum() / len(y)
