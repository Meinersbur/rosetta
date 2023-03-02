#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import unittest
from context import rosetta
from rosetta.stat import statistic
import  rosetta.stat  as stat





class StatisticTests(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(statistic([]).mean ,None)
        self.assertEqual(statistic([2,3,7]).mean ,4)

    def test_median(self):
        self.assertEqual(statistic([]).median ,None)
        self.assertEqual(statistic([2,4]).median ,3)
        self.assertEqual(statistic([2,3,7]).median ,3)

    def test_minimum(self):
        self.assertEqual(statistic([]).minimum ,None)
        self.assertEqual(statistic([-2, 2]).minimum ,-2)

    def test_maximum(self):
        self.assertEqual(statistic([]).maximum ,None)
        self.assertEqual(statistic([-2, 2]).maximum ,2)

    def test_variance(self):
        self.assertEqual(statistic([]).variance ,0)
        self.assertEqual(statistic([42]).variance,0)
        self.assertEqual(statistic([2, 6]).variance,4)




if __name__ == '__main__':
    unittest.main()
