#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import contextlib
import io
import re

from context import rosetta
import rosetta.driver
from rosetta.util.support import *


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.srcdir = mkpath(__file__ ).parent.parent.parent
        self.test_dir = tempfile.TemporaryDirectory(prefix='reporttest-')
        self.rootdir = mkpath( self.test_dir)
        print("srcdir: " , self.srcdir)
        print("rootdir: " , self.rootdir)


    def tearDown(self):
        self.test_dir.cleanup()
        



    def test_single(self):
        reportfile = self.rootdir / 'singleresult.html'
        rosetta.driver.driver_main(argv=[None,  '--use-results-rdir', mkpath(__file__ ).parent / 'resultfiles'/ 'single', '--reportfile',  reportfile], mode= rosetta.driver.DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir)   

        with reportfile.open('r') as f:
            s = f.read()
        self.assertRegex(s, r'Benchmark\ Report')
        self.assertRegex(s, r'idioms\.assign')
        self.assertRegex(s, r'serial')


    def test_multi_ppm(self):
        reportfile = self.rootdir / 'multiresult.html'
        rosetta.driver.driver_main(argv=[None,  '--use-results-rdir', mkpath(__file__ ).parent / 'resultfiles'/ 'multi_ppm', '--reportfile',  reportfile], mode= rosetta.driver.DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir)   

        with reportfile.open('r') as f:
            s = f.read()
        self.assertRegex(s, r'Benchmark\ Report')
        self.assertRegex(s, r'idioms\.assign')
        self.assertRegex(s, r'serial')
        self.assertRegex(s, r'cuda')



if __name__ == '__main__':
    unittest.main()
