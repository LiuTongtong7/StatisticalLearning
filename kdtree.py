#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 
# Created by liutongtong on 2019-04-12 21:50
#

"""This module implements KD Tree with Python3.

The algorithms are based on Chapter 3 of the book 统计学习方法 (李航, 2012).
"""

import numpy as np

from data_structure import PriorityQueue


class KDNode(object):
    """Node of KD Tree

    Attributes:
        depth (int): The depth of the node. The corresponding axis of the node is equal to depth mod dimensions.
        threshold (int): The threshold to split the data set into two halves. For leaf nodes, the threshold is None.
        indices (int): The indices of the data in the node. For non-leaf nodes, the indices are none.
        left_child (KDNode): The left child of the node.
        right_child (KDNode): The right child of the node.
    """

    def __init__(self, depth, threshold=None, indices=None, left_child=None, right_child=None):
        self.depth = depth
        self.threshold = threshold
        self.indices = indices
        self.left_child = left_child
        self.right_child = right_child


class KDTree(object):
    """Implementation of KD Tree.

    Attributes:
        print: Print the structure of the tree.
        query: Query the tree for the k nearest neighbors.
        query_radius: Query the tree for neighbors within a radius r.
    """

    def __init__(self, data, leaf_size=40, p=2):
        """Init KD Tree.

        Args:
            data (array-like, shape (n_samples, n_features)): Data set.
            leaf_size (int): Number of points at which to switch to brute-force.
            p (int): The order of the minkowski distance.
        """
        self.leaf_size = max(leaf_size, 1)
        self.p = max(p, 1)
        if data is not None:
            self.data = np.array(data)
            self.n_samples, self.n_features = np.shape(self.data)
            self.indices = np.arange(self.n_samples)
            self.root = self.__build(0, 0, self.n_samples - 1)
        else:
            self.root = None

    def __split(self, axis, left, right):
        """Reorder the data of a node and split them into two halves.

        The algorithm is similar to quick sort. The current node contains data[indices[left:right+1]] and
            the split accords to `axis` axis.
        """
        mid = left + ((right - left) >> 1)
        while True:
            pivot = self.data[self.indices[right]][axis]
            l, r = left, right
            while l < r:
                while self.data[self.indices[l]][axis] < pivot and l < r:
                    l += 1
                while self.data[self.indices[r]][axis] >= pivot and l < r:
                    r -= 1
                self.indices[[l, r]] = self.indices[[r, l]]
            self.indices[[l, right]] = self.indices[[right, l]]
            if l > mid:
                right = l - 1
            elif l < mid:
                left = l + 1
            else:
                break
        return mid

    def __build(self, depth, left, right):
        """Build the tree.

        The algorithm is recursive. The current node contains data[indices[left:right+1]] and
            the current depth is `depth`.
        """
        if right - left + 1 > self.leaf_size:
            axis = depth % self.n_features
            mid = self.__split(axis, left, right)
            node = KDNode(depth, threshold=self.data[self.indices[mid]][axis])
            node.left_child = self.__build(depth + 1, left, mid)
            node.right_child = self.__build(depth + 1, mid + 1, right)
        else:
            node = KDNode(depth, indices=self.indices[left:right+1])
        return node

    def print(self):
        """Print the structure of the tree."""
        def helper(node):
            if not node:
                return
            if node.left_child or node.right_child:
                helper(node.left_child)
                print('--' * node.depth + str(node.threshold))
                helper(node.right_child)
            else:
                print('--' * node.depth + str(node.indices))

        helper(self.root)

    def __get_distance(self, a, b):
        """Calculate the minkowski distance between points a and b."""
        return np.power(np.sum(np.power(a - b, self.p)), 1 / self.p)

    def query(self, x, k, return_distance=False, sort_results=False):
        """Query the tree for the k nearest neighbors.

        Args:
            x (array-like, shape (n_features,)): Point to query.
            k (int): The number of nearest neighbors to return.
            return_distance (bool): If True, return a tuple (i, d) of indices and distances. If False, return array i.
            sort_results (bool): If True, then indices and distances of each point are sorted on return, so that the
                first point is the closest one. Otherwise, neighbors are returned in an arbitrary order.

        Returns:
            ind: if return_distance == False
            (ind, dist): if return_distance == True
            ind (array of integers, shape (k,)): The list of indices of the neighbors of the corresponding point.
            dist (array of doubles, shape (k,)): The list of distances to the neighbors of the corresponding point.
        """
        if not self.root:
            return None
        queue = PriorityQueue(maxsize=k)

        def helper(node, x, queue):
            if node.indices is not None:  # 叶节点
                for idx in node.indices:
                    queue.push((-1 * self.__get_distance(x, self.data[idx]), idx))
                return -1 * queue.front()[0] if queue.full() else float('inf')
            else:
                if x[node.depth % self.n_features] <= node.threshold:
                    curr_child, next_child = node.left_child, node.right_child
                else:
                    curr_child, next_child = node.right_child, node.left_child
                max_dist = helper(curr_child, x, queue)
                if abs(x[node.depth % self.n_features] - node.threshold) < max_dist:
                    max_dist = helper(next_child, x, queue)
                return max_dist

        helper(self.root, x, queue)
        res = [(i, -d) for d, i in queue.queue]
        if sort_results:
            res.sort(key=lambda r: r[1])
        res_i = [i for i, _ in res]
        res_d = [d for _, d in res]
        return (res_i, res_d) if return_distance else res_i

    def query_radius(self, x, r, count_only=False, return_distance=False, sort_results=False):
        """Query the tree for neighbors within a radius r.

        Args:
            x (array-like, shape (n_features,)): Point to query.
            r (double): distance within which neighbors are returned
            count_only (bool): If True, return only the count of points within distance r. If False, return the indices
                (and distances) of all points within distance r
            return_distance (bool): If True, return a tuple (i, d) of indices and distances. If False, return array i.
            sort_results (bool): If True, then indices and distances of each point are sorted on return, so that the
                first point is the closest one. Otherwise, neighbors are returned in an arbitrary order.

        Returns:
            count: If count_only == True
            ind: If count_only == False and return_distance == False
            (ind, dist): If count_only == False and return_distance == True
            count (int): The number of neighbors within a distance r of the corresponding point.
            ind (array of integers, shape (k,)): The list of indices of the neighbors of the corresponding point
            dist (array of doubles, shape (k,)): The list of distances to the neighbors of the corresponding point
        """
        if not self.root:
            return None
        res = list()

        def helper(node, x, queue):
            if node.indices is not None:  # 叶节点
                for idx in node.indices:
                    dist = self.__get_distance(x, self.data[idx])
                    if dist <= r:
                        queue.append((idx, dist))
            else:
                if x[node.depth % self.n_features] <= node.threshold:
                    curr_child, next_child = node.left_child, node.right_child
                else:
                    curr_child, next_child = node.right_child, node.left_child
                helper(curr_child, x, queue)
                if abs(x[node.depth % self.n_features] - node.threshold) <= r:
                    helper(next_child, x, queue)

        helper(self.root, x, res)
        if count_only:
            return len(res)
        if sort_results:
            res.sort(key=lambda id: id[1])
        res_i = [i for i, _ in res]
        res_d = [d for _, d in res]
        return (res_i, res_d) if return_distance else res_i
