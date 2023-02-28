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




class UserBuilddirNinja(unittest.TestCase):
    def setUp(self):
        self.srcdir = mkpath(__file__ ).parent.parent.parent
        self.test_dir = tempfile.TemporaryDirectory(prefix='userninja-')
        self.builddir = mkpath(self.test_dir)
        print("srcdir: " , self.srcdir)
        print("builddir: " , self.builddir)
        invoke.diag('cmake', '-S', self.srcdir, '-B',self. builddir,  '-GNinja',  '-DCMAKE_BUILD_TYPE=Release','-DROSETTA_BENCH_FILTER=--filter=cholesky', cwd= self.builddir,onerror=invoke.Invoke.EXCEPTION)

    def tearDown(self):
        self.test_dir.cleanup()


    def test_build(self):
        rosetta.driver.driver_main( [None, '--build'], mode=DriverMode.USERBUILDDIR, builddir=self.builddir, srcdir=self.srcdir  )     
        self.assertTrue((self.builddir / 'benchmarks' / 'suites.polybench.cholesky.serial').exists())


if __name__ == '__main__':
    unittest.main()


