#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from context import rosetta
from rosetta.stat import statistic


class StatisticTests(unittest.TestCase):
    def test_count(self):
        self.assertEqual(statistic([]).count, 0)
        self.assertEqual(statistic([2, 3, 7]).count, 3)

    def test_mean(self):
        self.assertEqual(statistic([]).mean, None)
        self.assertEqual(statistic([42]).mean, 42)
        self.assertEqual(statistic([2, 3, 7]).mean, 4)

    def test_median(self):
        self.assertEqual(statistic([]).median, None)
        self.assertEqual(statistic([2, 4]).median, 3)
        self.assertEqual(statistic([2, 3, 7]).median, 3)

    def test_minimum(self):
        self.assertEqual(statistic([]).minimum, None)
        self.assertEqual(statistic([-2, 2]).minimum, -2)

    def test_maximum(self):
        self.assertEqual(statistic([]).maximum, None)
        self.assertEqual(statistic([-2, 2]).maximum, 2)

    def test_variance(self):
        self.assertEqual(statistic([]).variance, 0)
        self.assertEqual(statistic([42]).variance, 0)
        self.assertEqual(statistic([2, 6]).variance, 4)

    def test_stddev(self):
        self.assertEqual(statistic([]).stddev, 0)
        self.assertEqual(statistic([42]).stddev, 0)
        self.assertEqual(statistic([-1, 1]).stddev, 1)
        self.assertEqual(statistic([2, 6]).stddev, 2)
        self.assertEqual(statistic([0, 0, 6, 6]).stddev, 3)
        self.assertAlmostEqual(statistic([0, 0, 0, 0, 3.5, 7, 7, 7, 7]).stddev, 3.2998316455372216)

    def test_range(self):
        self.assertEqual(statistic([]).range, 0)
        self.assertEqual(statistic([42]).range, 0)
        self.assertEqual(statistic([-1, 1]).range, 2)
        self.assertEqual(statistic([2, 7]).range, 5)

    def test_abserr(self):
        self.assertEqual(statistic([]).abserr(), None)
        self.assertEqual(statistic([42]).abserr(), None)
        self.assertAlmostEqual(statistic([0, 1]).abserr(), 4.492321766137882)
        self.assertAlmostEqual(statistic([-1, 1]).abserr(), 8.984643532275763)
        self.assertAlmostEqual(
            statistic([0, 0, 6, 6]).abserr(), 4.7736694579263945
        )  # Should be 3 plusminus 4.77367 according to wolframalpha
        self.assertAlmostEqual(
            statistic([0, 0, 0, 0, 3.5, 7, 7, 7, 7]).abserr(), 2.5364751398409346
        )  # Should be 3.5 plusminus ~2.7


if __name__ == '__main__':
    unittest.main()
