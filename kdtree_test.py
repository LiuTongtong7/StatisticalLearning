#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 
# Created by liutongtong on 2019-04-12 21:50
#

import numpy as np
import unittest

from kdtree import KDTree


class TestKDTree(unittest.TestCase):

    def setUp(self):
        self.tree1 = KDTree([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]], leaf_size=1)
        self.tree2 = KDTree([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]], leaf_size=2)

    def test_build(self):
        self.tree1.print()
        self.tree2.print()

    def test_query(self):
        self.assertTrue(np.array_equal(self.tree1.query([4, 5], 1), [1]))
        self.assertTrue(np.array_equal(self.tree1.query([4, 5], 2), [3, 1]))
        ind, dist = self.tree2.query([4, 5], 1, return_distance=True)
        self.assertTrue(np.array_equal(ind, [1]))
        self.assertEqual(dist, [np.sqrt(2)])
        self.assertTrue(np.array_equal(self.tree2.query([4, 5], 2, sort_results=True), [1, 3]))

    def test_query_radius(self):
        self.assertTrue(np.array_equal(self.tree1.query_radius([7, 4], 3), [2, 5, 1]))
        self.assertTrue(np.array_equal(self.tree1.query_radius([7, 4], 3, count_only=True), 3))
        ind, dist = self.tree2.query_radius([7, 4], 3, return_distance=True)
        self.assertTrue(np.array_equal(ind, [2, 5, 1]))
        self.assertEqual(dist, [np.sqrt(8), 2, 2])
        self.assertTrue(np.array_equal(self.tree2.query_radius([7, 4], 3, sort_results=True), [5, 1, 2]))


if __name__ == '__main__':
    unittest.main()
