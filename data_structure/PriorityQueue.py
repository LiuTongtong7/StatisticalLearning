#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 
# Created by liutongtong on 2019-04-16 23:14
#

"""This module implements priority queue with Python3.

The priority queue is based on a min heap and implemented with the package `heapq`.
"""

import heapq


class PriorityQueue(object):
    """Implementation of priority queue.

    Attributes:
        queue (list): Elements of the queue.
        size (int): The number of elements in the queue.
        maxsize (int): The maximum number of elements in the queue.

        push: Add an element into the queue.
        pop: Remove and return an element from the queue.
        front: Return the minimum element of the queue.
        empty: Return whether the queue is empty.
        full: Return whether the queue is full.
    """

    def __init__(self, maxsize=0):
        """Init priority queue.

        Args:
            maxsize (int): The maximum number of elements in the queue.
                If maxsize is less than or equal to zero, the queue size is infinite.
        """
        self.queue = list()
        self.size = 0
        self.maxsize = maxsize if maxsize > 0 else float('inf')

    def push(self, element):
        """Add an element into the queue.

        If the queue is full and the element is greater than the root, remove the root and add the element.

        Args:
            element (objects): The element to add.

        Returns:
            bool: True if the element is added into the queue, False otherwise.
        """
        if not self.full():
            heapq.heappush(self.queue, element)
            self.size += 1
            return True
        else:
            if element >= self.queue[0]:
                heapq.heapreplace(self.queue, element)
                return True
            else:
                return False

    def pop(self):
        """Remove and return an element from the queue.

        Returns:
            object: The root element if the queue is not empty, None otherwise.
        """
        if not self.empty():
            self.size -= 1
            return heapq.heappop(self.queue)
        else:
            return None

    def front(self):
        """Return the minimum element of the queue.

        Returns:
            objects: The root element if the queue is not empty, None otherwise.
        """
        return self.queue[0] if not self.empty() else None

    def empty(self):
        """Return whether the queue is empty.

        Returns:
            bool: True if the queue is empty, False otherwise.
        """
        return self.size == 0

    def full(self):
        """Return whether the queue is full.

        Returns:
            bool: True if the queue is full, False otherwise.
        """
        return self.size >= self.maxsize
