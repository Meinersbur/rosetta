#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract the unittest.TestCase names so CTest can run them independently
"""


import os
import unittest
import collections

scriptdir = os.path.dirname(__file__)


def generate_test_cases(suite):
    for test in suite:
        if isinstance(test, unittest.TestCase):
            yield test
        else:
            for test_case in generate_test_cases(test):
                yield test_case

def generate_test_cases(suite):
    for test in suite:
        if isinstance(test, unittest.TestCase):
            yield test
        else:
            for test_case in generate_test_cases(test):
                yield test_case

loader = unittest.TestLoader()
suite = loader.discover(scriptdir, pattern='test_*.py')



testclasses = collections.  defaultdict(list)
for s in generate_test_cases(suite):
        tm = getattr(s, s._testMethodName)
        testId = s.id()
        if testId.startswith("unittest.loader._FailedTest"):
            #loader_errors.append(s._exception)
            pass
        else:
            modulename,classname,_ = testId.split('.',maxsplit=2)
            testclasses[f"{modulename}.{classname}"].append(s)


for k,v in testclasses.items():
    print(k,end=' ')

print()