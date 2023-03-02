#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from context import rosetta
import unittest
import sys
import tempfile
from  rosetta.driver import *
from rosetta.util.support import *





class ManagedBuilddirTests(unittest.TestCase):
    def setUp(self):
        rosetta.runner.reset_registered_benchmarks()

        self.srcdir = mkpath(__file__ ).parent.parent.parent
        self.test_dir = tempfile.TemporaryDirectory(prefix='managed-')
        self.rootdir = mkpath( self.test_dir)
        print("srcdir: " , self.srcdir)
        print("rootdir: " , self.rootdir)

    def tearDown(self):
        self.test_dir.cleanup()



    def test_clean(self):
        builddir = (self.rootdir / 'build' / 'build-somebuild')
        builddir.mkdir(exist_ok=True,parents=True)
        with (builddir / 'CMakeCache.txt').open('w+') as f:
            print("Not a real CMakeCache.txt; for testing only",file=f)

        rosetta.driver.driver_main( argv=[None, '--clean'], mode=DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir  )     
        builds = list((self.rootdir / 'build' ).iterdir())
        self.assertEquals(  len( builds), 0 )


    def test_configure(self):
        rosetta.driver.driver_main(  argv=[None, '--configure', "--compiler-arg=O2:-O2",  "--compiler-arg=O3:-O3"], mode=DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir  )     
        builds = list((self.rootdir / 'build' ).iterdir())
        self.assertEquals({ b.name for b in builds }, {'build-O2', 'build-O3'}  )
        for build in builds:
            self.assertTrue((build / 'CMakeCache.txt').exists())


    def test_build(self):
        rosetta.driver.driver_main(  argv=[None, '--build', "--compiler-arg=O2:-O2",  "--compiler-arg=O3:-O3", "--cmake-def=ROSETTA_BENCH_FILTER=--filter=cholesky"], mode=DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir  )     
        for build in (self.rootdir / 'build' ).iterdir():
            self.assertTrue((build / 'benchmarks' / 'Release' / 'suites.polybench.cholesky.serial').exists())


    def test_verify(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(Tee( f, sys.stdout)):
            rosetta.driver.driver_main( argv= [None, '--verify',  "--cmake-def=ROSETTA_BENCH_FILTER=--filter=idioms.assign", "--compiler-arg=O3:-O3"], mode=DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir  )     

        s = f.getvalue()
        self.assertTrue(re.search(r'^Output of idioms\.assign\..* considered correct$',s, re.MULTILINE ))
        self.assertFalse(re.search(r'^Array data mismatch\:',s, re.MULTILINE));

     


    def test_bench(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(Tee( f, sys.stdout)):
            rosetta.driver.driver_main( argv= [None, '--bench',  "--cmake-def=ROSETTA_BENCH_FILTER=--filter=cholesky"], mode=DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir  )     
    
        # Evaluate output
        s=f.getvalue()
        self. assertTrue(re.search(r'Benchmark.+Wall', s, re.MULTILINE), "Evaluation table Header")
        self. assertTrue(re.search(r'suites\.polybench\.cholesky', s, re.MULTILINE), "Benchmark entry")
    
        # Check benchmarking results
        results = list((self.rootdir /'results').glob('**/*.xml'))
        self.assertTrue(len(results)>=1)
        for r in results:
            self.assertTrue(r.name.startswith('suites.polybench.cholesky.'), "Must only run filtered tests" )

        # Check report
        reports = list((self.rootdir /'results').glob('report_*.html'))
        self.assertEqual(len(reports),1)


    def test_compare(self):
        rosetta.driver.driver_main( argv= [None, '--bench', '--compare',  "--cmake-def=ROSETTA_BENCH_FILTER=--filter=cholesky",  "--compiler-arg=O2:-O2", "--compiler-arg=O3:-O3"], mode=DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir  )     
        
        # Check result dir content
        resultlist = list((self.rootdir / 'build' ).iterdir())
        self.assertTrue(len(resultlist)== 2, "resultdir for xml files and the html report")

        # Check result files
        results = list((self.rootdir /'results').glob('**/*.xml'))
        self.assertTrue(len(results)>=2, "At least one serial result per optimization level")
        for r in results:
            self.assertTrue(r.name.startswith('suites.polybench.cholesky.'), "Must only run filtered tests" )

        # Check report
        reports = list((self.rootdir /'results').glob('report_*.html'))
        self.assertEqual(len(reports),1)


if __name__ == '__main__':
    unittest.main()
