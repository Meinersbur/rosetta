#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from context import rosetta
import unittest
import os
import sys
import tempfile
from  rosetta.driver import *
from rosetta.util.support import *



class ManagedBuilddirTests(unittest.TestCase):
    def setUp(self):
        self.srcdir = mkpath(__file__ ).parent.parent.parent
        self.test_dir = tempfile.TemporaryDirectory('managed-')
        self.rootdir = mkpath( self.test_dir)
        print("srcdir: " , self.srcdir)
        print("rootdir: " , self.rootdir)

    def tearDown(self):
        self.test_dir.cleanup()



    def test_clean(self):
        builddir = (self.rootdir / 'build' / 'build-somebuild')
        builddir.mkdir(exist_ok=True,parents=True)
        with (builddir / 'CMakeCache.txt').open('w+') as f:
            print("dummy file",file=f)

        rosetta.driver.driver_main( [None, '--clean'], mode=DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir  )     
        builds = list((self.rootdir / 'build' ).iterdir())
        self.assertEquals(  len( builds), 0 )

    def test_configure(self):
        rosetta.driver.driver_main( [None, '--configure', "--compiler-arg=O2:-O2",  "--compiler-arg=O3:-O3"], mode=DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir  )     
        builds = list((self.rootdir / 'build' ).iterdir())
        self.assertEquals({ b.name for b in builds }, {'build-O2', 'build-O3'}  )
        for build in builds:
            self.assertTrue((build / 'CMakeCache.txt').exists())

    def test_build(self):
        rosetta.driver.driver_main( [None, '--build', "--compiler-arg=O2:-O2",  "--compiler-arg=O3:-O3", "--cmake-def=ROSETTA_BENCH_FILTER=--filter=cholesky"], mode=DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir  )     
        for build in (self.rootdir / 'build' ).iterdir():
            self.assertTrue((build / 'benchmarks' / 'Release' / 'suites.polybench.cholesky.serial').exists())

    #def test_probe(self):
    #    rosetta.driver.driver_main( [None, '--probe',  "--cmake-def=ROSETTA_BENCH_FILTER=--filter=cholesky", "--compiler-arg=O3:-O3",   "--limit-walltime=1s"], mode=DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir  )     

    def test_verify(self):
        rosetta.driver.driver_main( [None, '--verify',  "--cmake-def=ROSETTA_BENCH_FILTER=--filter=cholesky", "--compiler-arg=O3:-O3"], mode=DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir  )     

    def test_bench(self):
        rosetta.driver.driver_main( [None, '--bench',  "--cmake-def=ROSETTA_BENCH_FILTER=--filter=cholesky", "--compiler-arg=O3:-O3"], mode=DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir  )     


if __name__ == '__main__':
    unittest.main()
