# -*- coding: utf-8 -*-

import cwcwidth
import importlib.util
import importlib
import contextlib
import typing
import configparser
import io
from collections import defaultdict
import math
import colorama
import xml.etree.ElementTree as et
from typing import Iterable
import json
import datetime
import os
import pathlib
import subprocess
import argparse
import sys
from itertools import count
from .util.cmdtool import *
from .util.orderedset import OrderedSet
from .util.support import *
from .util import invoke

# Summary statistics
# Don't trust yet, have to check correctness
# TODO: Use numpy
class Statistic:
    # TODO: Consider using SciPy's scipy.distributions.t.cdf if available
    # Or just directly use scipy.distributions.ttest_ind
    # https://pythonguides.com/scipy-confidence-interval/
    studentt_density_95 = list({
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
        35: 2.030,
        40: 2.021,
        45: 2.014,
        50: 2.009,
        60: 2.000,
        70: 1.994,
        80: 1.990,
        90: 1.987,
        100: 1.984,
        150: 1.976,
        200: 1.972,
        250: 1.969,
        300: 1.968,
        400: 1.966,
        500: 1.965,
        600: 1.964,
        800: 1.963,
        1000: 1.962,
        100000: 1.960,
    }.items())

    def __init__(self, samples, sum, sumabs, sumsqr, sumreciproc, geomean):
        self._samples = list(samples)  # Assumed sorted
        self._sum = sum
        self._sumabs = sumabs
        self._sumsqr = sumsqr
        self._sumreciproc = sumreciproc
        self._geomean = geomean

    @property
    def samples(self):
        return self._samples

    @property
    def is_empty(self):
        return not not self._samples

    @property
    def count(self):
        return len(self._samples)

    @property
    def minimum(self):
        return self._samples[0]

    @property
    def maximum(self):
        return self._samples[-1]

    # Location

    @property
    def mean(self):
        return self._sum / self.count

    @property
    def geomean(self):
        return self._geomean

    @property
    def harmonicmean(self):
        return self.count / self._sumreciproc

    @property
    def median(self):
        return self.quantile(1, 2)

    def quantile(self, k, d):
        assert d >= 1
        assert k >= 0 <= d

        if not self._vals:
            return None

        if k == 0:
            return self._vals[0]
        if k == n:
            return self._vals[-1]

        n = self.count
        if (k * n - 1) % d == 0:
            return self._vals[(k * n - 1) // d]

    def quartile(self, k: int):
        return self.quantile(k, 4)

    def decile(self, k: int):
        return self.quantile(k, 10)

    def percentile(self, k: int):
        return self.quantile(k, 100)

    @property
    def midrange(self):
        return (self.minimum + self.maximum) / 2

    def mode(self, boxsize):
        boxes = defaultdict(0)
        for v in self._samples:
            boxidx = round(v / boxsize)
            boxes[boxidx] += 1

        maxcount = 0
        boxids = []
        for boxidx, count in boxes.items():
            if count < maxcount:
                continue
            if count > maxcount:
                maxcount = count
                boxids.clear()
                continue
            boxids.append(boxids)

        if len(boxids) % 2 == 0:
            midboxid = (boxids[len(boxids) // 2] + boxids[len(boxids) // 2 + 1]) / 2
        else:
            midboxid = boxids[len(boxids) // 2]
        return midboxid * boxsize

    # Spread

    @property
    def variance(self):
        def sqr(x):
            return x * x
        n = self.count
        return self._sumsqr / n - sqr(self.mean)

    @property
    def corrected_variance(self):
        n = self.count
        return self.variance * n / (n - 1)

    @property
    def stddev(self):
        return math.sqrt(self.variance)

    @property
    def corrected_stddev(self):
        return math.sqrt(self.corrected_variance)

    @property
    def relative_stddev(self):
        return self.stddev / abs(self.mean)

    @property
    def range(self):
        return self.maximum - self.minimum

    # Mean squared error/deviation
    @property
    def mse(self):
        def sqr(x):
            return x * x
        e = self.median  # Using median as estimator
        return sum(sqr(x - e) for x in self._vals) / self.count

    # Root mean squared error/deviation
    @property
    def rmse(self):
        return math.sqrt(self.mse)

    @property
    def relative_rmse(self):
        return self.rmse / abs(self.median)

    # Mean absolute error/deviation
    @property
    def mad(self):
        median = self.median
        return sum(abs(x - median) for x in self._vals) / self.count

    # Symmetric confidence interval around mean, assuming normal distributed samples
    # TODO: Asymetric confidence interval
    def abserr(self, ratio=0.95):
        assert ratio == 0.95, r"Only supporting two-sided 95% confidence interval"
        n = self.count
        if n < 2:
            return None

        # Table lookup
        # TODO: bisect
        c = None
        for (n1, p1), (n2, p2) in zip(Statistic.studentt_density_95, Statistic.studentt_density_95[1:]):
            if n1 <= n <= n2:
                # Linear interpolation
                r = (n - n1) / (n2 - n1)
                c = r * p2 + (1 - r) * p1
                break
        if c is None:
            c = Statistic.studentt_density_95[-1][1]

        return c * self.corrected_variance / math.sqrt(n)

    def relerr(self, ratio=0.95):
        mean = self.mean
        if not mean:
            return None
        return self.abserr(ratio=ratio) / self.mean

    # TODO: signal/noise ratio (relative_rmse?)


def statistic(data):
    vals = sorted(d for d in data if d is not None)
    n = 0
    hasnonpos = False
    sum = 0
    sumabs = 0
    sumsqr = 0
    sumreciproc = 0
    prod = 1
    for v in vals:
        if v <= 0:
            hasnonpos = True
        sum += v
        sumabs = abs(v)
        sumsqr += v * v
        if not hasnonpos:
            sumreciproc += 1 // v
            prod *= v
        n += 1

    if hasnonpos:
        geomean = None
        sumreciproc = None
    else:
        geomean = prod**(1 / n)

    return Statistic(samples=vals, sum=sum, sumabs=sumabs, sumsqr=sumsqr, geomean=geomean, sumreciproc=sumreciproc)
