#! /usr/bin/env python3
# -*- coding: utf-8 -*-


from context import rosetta
import unittest
import tempfile
import  rosetta.util.invoke as invoke
import pathlib
from rosetta.util.support import *
import rosetta.runner
from context import rosetta
import unittest
import os
import sys
import tempfile
from  rosetta.driver import *
from rosetta.util.support import *




class UserBuilddirMulticonfig(unittest.TestCase):
    def setUp(self):
        self.srcdir = mkpath(__file__ ).parent.parent.parent
        self.test_dir = tempfile.TemporaryDirectory(prefix='userninja-')
        self.builddir = mkpath(self.test_dir)
        print("srcdir: " , self.srcdir)
        print("builddir: " , self.builddir)
        invoke.diag('cmake', '-S', self.srcdir, '-B',self. builddir,  '-GNinja Multi-Config',  '-DROSETTA_BENCH_FILTER=--filter=cholesky', cwd= self.builddir,onerror=invoke.Invoke.EXCEPTION)

        self.benchlistfile = self.builddir / 'benchmarks' / 'benchlist.py'


    def tearDown(self):
        self.test_dir.cleanup()


    def test_build(self):
        rosetta.driver.driver_main( [None, '--build'], mode=DriverMode.USERBUILDDIR, benchlistfile=  self.benchlistfile, builddir=self.builddir, srcdir=self.srcdir  )     
        self.assertTrue((self.builddir / 'benchmarks' / 'Release' / 'suites.polybench.cholesky.serial').exists())


    def test_bench(self):
        rosetta.driver.driver_main([None, '--bench'], mode=DriverMode.USERBUILDDIR,benchlistfile=  self.benchlistfile, builddir=self.builddir, srcdir=self.srcdir)     
        results = list((self.builddir /'results').glob('**/*.xml'))
        self.assertTrue(len(results)>=1)
        for r in results:
            self.assertTrue(r.name.startswith('suites.polybench.cholesky.'), "Must only run filtered tests" )





if __name__ == '__main__':
    unittest.main()


