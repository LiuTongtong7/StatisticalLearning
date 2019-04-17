#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 
# Created by liutongtong on 2019-04-16 23:08
#

"""This module implements K-Nearest Neighbor with Python3.

The algorithms are based on Chapter 3 of the book 统计学习方法 (李航, 2012).

TODO:
    * Add Ball Tree.
"""

import numpy as np

from kdtree import KDTree


class KNeighborsClassifier(object):
    """Implementation of KNeighborsClassifier.

    Attributes:
        fit: Fit the model (build tree based on X).
        kneighbors: Find the K-neighbors of a point.
        predict_proba: Return probability estimates for the test data X.
        predict: Predict the class labels for the provided data.
        score: Returns the mean accuracy on the given test data and labels.
    """

    def __init__(self, n_neighbors=5, weights='uniform', algorithm='kd_tree', leaf_size=30, p=2):
        """Init KNeighborsClassifier.

        Args:
            n_neighbors (int): Number of neighbors to use by default for kneighbors queries.
            weights ({'uniform', 'distance'}): Weight function used in prediction. Possible values:
                'uniform': uniform weights. All points in each neighborhood are weighted equally.
                'distance': weight points by the inverse of their distance.
            algorithm ({'kd_tree', 'ball_tree'}): Algorithm used to compute the nearest neighbors.
            leaf_size (int): Leaf size passed to BallTree or KDTree.
            p (int): Power parameter for the Minkowski metric.
        """
        self.n_neighbors = max(n_neighbors, 1)
        self.weights = weights if weights in ['uniform', 'distance'] else 'uniform'
        self.algorithm = algorithm if algorithm in ['kd_tree'] else 'kd_tree'
        self.leaf_size = leaf_size
        self.p = p

        self.tree = None
        self.X, self.y = None, None

    def fit(self, X, y):
        """Fit the model (build tree based on X).

        Args:
            X (array-like, shape (n_samples, n_features)): Training data.
            y (array, shape (n_samples,)): Target values.
        """
        self.X, self.y = np.array(X), np.array(y)
        if self.algorithm == 'kd_tree':
            self.tree = KDTree(X, self.leaf_size, self.p)
        if self.algorithm == 'ball_tree':
            pass

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Find the K-neighbors of a point.

        Args:
            X (array-like, shape (n_query, n_features)): The query point or points.
            n_neighbors (int): Number of neighbors to get (default is the value passed to the constructor).
            return_distance (bool): If False, distances will not be returned.

        Returns:
            ind (array of integers, shape (n_query, n_neighbors)): The list of indices of the neighbors of the
                corresponding point
            dist (array of doubles, shape (n_query, n_neighbors)): The list of distances to the neighbors of the
                corresponding point.
        """
        n_neighbors = n_neighbors if n_neighbors is not None else self.n_neighbors
        res = list()
        for x in np.array(X):
            res.append(self.tree.query(x, n_neighbors, return_distance=return_distance))
        return res

    @staticmethod
    def __get_proba(labels, weights):
        """Calculate probability for each label."""
        proba = dict()
        for label, weight in zip(labels, weights):
            proba.setdefault(label, 0)
            proba[label] += weight
        total_weights = sum(proba.values())
        for label in proba:
            proba[label] /= total_weights
        return proba

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Args:
            X (array-like, shape (n_query, n_features)): Query samples.

        Returns:
            p (array, shape (n_query, dict{label: proba})): The class probabilities of the query samples.
        """
        res = list()
        for x in np.array(X):
            neighbors = self.tree.query(x, self.n_neighbors, return_distance=True)
            labels = self.y[[i for i, _ in neighbors]]
            weights = np.ones(len(neighbors)) if self.weights == 'uniform' else 1 / np.array([d for _, d in neighbors])
            res.append(self.__get_proba(labels, weights))
        return res

    def predict(self, X):
        """Predict the class labels for the provided data.

        Args:
            X (array-like, shape (n_query, n_features)): Query samples.

        Returns:
            y (array, shape (n_query,)): Class labels for each query sample.
        """
        probas = self.predict_proba(X)
        res = list()
        for proba in probas:
            label, maxp = None, 0
            for l in proba:
                if proba[l] > maxp:
                    label, maxp = l, proba[l]
            res.append(label)
        return res

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X (array-like, shape (n_query, n_features)): Test samples.
            y (array, shape (n_query,)): True labels for X

        Returns:
            score (float): Mean accuracy of self.predict(X) wrt. y.
        """
        return (self.predict(X) == np.array(y)).sum() / len(y)
