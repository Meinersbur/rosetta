#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import contextlib
import io
import re

from context import rosetta
import rosetta.driver
from rosetta.util.support import *


class EvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srcdir = mkpath(__file__ ).parent.parent.parent
        cls.test_dir = tempfile.TemporaryDirectory(prefix='managed-')
        cls.rootdir = mkpath( cls.test_dir)
        print("srcdir: " , cls.srcdir)
        print("rootdir: " , cls.rootdir)

    @classmethod
    def tearDownClass(cls):
        cls.test_dir.cleanup()
        
    def tearDown(self):
        # Just for evaluating, the rootdir shouldn't be used
        # Data from --use-results-rdir instead
        rootcontent = list(self.rootdir .iterdir())
        self.assertEquals(len(rootcontent),0)


    def test_singleresult(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(Tee( f, sys.stdout)):
            rosetta.driver.driver_main(argv=[None, '--evaluate', '--use-results-rdir', mkpath(__file__ ).parent / 'resultfiles'/ 'single' ], mode= rosetta.driver.DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir)     

        s=f.getvalue()
        self. assertTrue(re.search(r'Benchmark.+Wall', s, re.MULTILINE), "Evaluation table Header")
        self. assertTrue(re.search(r'suites\.polybench\.atax', s, re.MULTILINE), "Benchmark entry")


if __name__ == '__main__':
    unittest.main()
