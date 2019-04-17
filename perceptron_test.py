#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 
# Created by liutongtong on 2019-03-26 00:44
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unittest

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from perceptron import Perceptron


class TestPerceptron(unittest.TestCase):

    def setUp(self):
        self.iris = load_iris()
        self.df = pd.DataFrame(self.iris.data, columns=self.iris.feature_names)
        self.df['label'] = self.iris.target

        # plt.scatter(self.df.loc[self.df['label'] == 0, self.iris.feature_names[0]],
        #             self.df.loc[self.df['label'] == 0, self.iris.feature_names[1]], label='0')
        # plt.scatter(self.df.loc[self.df['label'] == 1, self.iris.feature_names[0]],
        #             self.df.loc[self.df['label'] == 1, self.iris.feature_names[1]], label='1')
        # plt.xlabel(self.iris.feature_names[0])
        # plt.ylabel(self.iris.feature_names[1])
        # plt.legend()
        # plt.show()

        X = self.df.loc[self.df['label'] <= 1, self.iris.feature_names[0:2]]
        y = self.df.loc[self.df['label'] <= 1, 'label'] * 2 - 1
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=311)

    def test_fit(self):
        estimator = Perceptron(eta0=0.5, random_state=311)
        estimator.fit(self.X_train, self.y_train)
        w, b = estimator.coef_, estimator.intercept_
        print(f"w: {w}, b: {b}, n_iter: {estimator.n_iter_}")

        x_data = np.linspace(4, 7, 10)
        y_data = -(w[0] * x_data + b) / w[1]
        plt.plot(x_data, y_data, label='perceptron')
        plt.scatter(self.X_train.loc[self.y_train == -1, self.iris.feature_names[0]],
                    self.X_train.loc[self.y_train == -1, self.iris.feature_names[1]], label='-1')
        plt.scatter(self.X_train.loc[self.y_train == 1, self.iris.feature_names[0]],
                    self.X_train.loc[self.y_train == 1, self.iris.feature_names[1]], label='1')
        plt.xlabel(self.iris.feature_names[0])
        plt.ylabel(self.iris.feature_names[1])
        plt.legend()
        plt.show()

    def test_predict(self):
        estimator = Perceptron(eta0=0.5, random_state=311)
        estimator.fit(self.X_train, self.y_train)
        self.assertEqual(estimator.score(self.X_test, self.y_test), 1)


if __name__ == '__main__':
    unittest.main()
