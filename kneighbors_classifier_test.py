#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 
# Created by liutongtong on 2019-04-16 23:08
#

import pandas as pd
import unittest

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from kneighbors_classifier import KNeighborsClassifier


class TestKNeighborClassifier(unittest.TestCase):

    def setUp(self):
        self.iris = load_iris()
        self.df = pd.DataFrame(self.iris.data, columns=self.iris.feature_names)
        self.df['label'] = self.iris.target

        X = self.df.loc[self.df['label'] <= 1, self.iris.feature_names[0:2]]
        y = self.df.loc[self.df['label'] <= 1, 'label'] * 2 - 1
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=311)

    def test_predict(self):
        estimator = KNeighborsClassifier(leaf_size=5)
        estimator.fit(self.X_train, self.y_train)
        self.assertEqual(estimator.score(self.X_test, self.y_test), 1)


if __name__ == '__main__':
    unittest.main()
