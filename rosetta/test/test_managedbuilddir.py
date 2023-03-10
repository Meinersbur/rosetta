#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import sys
import io
import tempfile

from context import rosetta
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












class ManagedBuilddirDefaultconfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srcdir = mkpath(__file__ ).parent.parent.parent
        cls.test_dir = tempfile.TemporaryDirectory(prefix='managed-')
        cls.rootdir = mkpath( cls.test_dir)
        print("srcdir: " , cls.srcdir)
        print("rootdir: " , cls.rootdir)

        rosetta.driver.driver_main(argv=[None, '--configure', '--cmake-def=ROSETTA_BENCH_FILTER=--filter=idioms.assign'], mode=DriverMode.MANAGEDBUILDDIR, rootdir=cls.rootdir, srcdir=cls.srcdir  )     

        cls.resultsdir= cls.rootdir / 'results'


    @classmethod
    def tearDownClass(cls):
        cls.test_dir.cleanup()


    def setUp(self):
        rosetta.runner.reset_registered_benchmarks()
        if self.resultsdir.exists():
            shutil.rmtree(self.resultsdir)



    def test_configure(self):
        builds = list((self.rootdir / 'build' ).iterdir())
        self.assertSetEqual({ b.name for b in builds }, {'defaultbuild'}  )
        self.assertTrue((self.rootdir / 'build' / 'defaultbuild' / 'CMakeCache.txt').exists())


    def test_build(self):
        rosetta.driver.driver_main(  argv=[None, '--build'], mode=DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir  )   
        buildlist = list( (self.rootdir / 'build' ).iterdir())
        self.assertEquals(len(buildlist),1)  
        self.assertTrue((self.rootdir / 'build' / 'defaultbuild' / 'benchmarks' / 'Release' / 'idioms.assign.serial').exists())




    def test_probe(self):
        problemsizefile = self.resultsdir /  'proberesult.ini'
        rosetta.driver.driver_main(  argv=[None, '--problemsizefile-out', problemsizefile, '--limit-walltime=10ms'],  default_action=DefaultAction.PROBE, mode=DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir  )       

        with problemsizefile.open('r') as f:
            s = f.read()

        self.assertRegex(s, r'\[idioms\.assign\]\nn\=')



    def test_verify(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(Tee( f, sys.stdout)):
            rosetta.driver.driver_main( argv= [None, '--verify'], mode=DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir  )     

        s = f.getvalue()
        self.assertTrue(re.search(r'^Output of .*idioms\.assign\..* considered correct$',s, re.MULTILINE ))
        self.assertFalse(re.search(r'^Array data mismatch\:',s, re.MULTILINE));


    def test_bench(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(Tee( f, sys.stdout)):
            rosetta.driver.driver_main(argv= [None, '--bench'], mode=DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir  )     
    
        # Evaluate output
        s=f.getvalue()
        self. assertTrue(re.search(r'Benchmark.+Wall', s, re.MULTILINE), "Evaluation table Header")
        self. assertTrue(re.search(r'idioms\.assign', s, re.MULTILINE), "Benchmark entry")
    
        # Check benchmarking results
        results = list((self.rootdir /'results').glob('**/*.xml'))
        self.assertTrue(len(results)>=1)
        for r in results:
            self.assertTrue(r.name.startswith('idioms.assign.'), "Must only run filtered tests" )

        # Check report
        reports = list((self.rootdir /'results').glob('**/report.html'))
        self.assertEqual(len(reports),1)




class ManagedBuilddirMulticonfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srcdir = mkpath(__file__ ).parent.parent.parent
        cls.test_dir = tempfile.TemporaryDirectory(prefix='managed-')
        cls.rootdir = mkpath( cls.test_dir)
        print("srcdir: " , cls.srcdir)
        print("rootdir: " , cls.rootdir)

        rosetta.driver.driver_main(argv=[None, '--configure', '--cmake-def=ROSETTA_BENCH_FILTER=--filter-include=idioms.assign --filter-include=suites.polybench.cholesky',  "--compiler-arg=O2:-O2",  "--compiler-arg=O3:-O3"], mode=DriverMode.MANAGEDBUILDDIR, rootdir=cls.rootdir, srcdir=cls.srcdir  )     

        cls.resultsdir= cls.rootdir / 'results'


    @classmethod
    def tearDownClass(cls):
        cls.test_dir.cleanup()


    def setUp(self):
        rosetta.runner.reset_registered_benchmarks()
        if self.resultsdir.exists():
            shutil.rmtree(self.resultsdir)




    def test_configure(self):
        builds = list((self.rootdir / 'build' ).iterdir())
        self.assertSetEqual({ b.name for b in builds }, {'build-O2', 'build-O3'}  )
        for build in builds:
            self.assertTrue((build / 'CMakeCache.txt').exists())



    def test_build(self):
        rosetta.driver.driver_main(  argv=[None, '--build', '--config=O2', '--config=O3'], mode=DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir  )     
        buildlist = list((self.rootdir / 'build' ).iterdir())
        self.assertEquals(len(buildlist),2)
        for build in buildlist:
            self.assertTrue((build / 'benchmarks' / 'Release' / 'idioms.assign.serial').exists())


    def test_bench(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(Tee( f, sys.stdout)):
            rosetta.driver.driver_main(argv= [None, '--bench', '--config=O2', '--config=O3'], mode=DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir  )     
    
        # Evaluate output
        s=f.getvalue()
        self. assertTrue(re.search(r'Benchmark.+PPM.*Wall', s, re.MULTILINE), "Evaluation table Header")
        self. assertTrue(re.search(r'O2.+O3', s, re.MULTILINE), "Comparison table Header")
        self. assertTrue(re.search(r'idioms\.assign.+serial', s, re.MULTILINE), "Benchmark entry")
        self. assertTrue(re.search(r'suites\.polybench\.cholesky.+serial', s, re.MULTILINE), "Benchmark entry")

        # Check benchmarking results
        results = list((self.rootdir /'results').glob('**/*.xml'))
        self.assertTrue(len(results)>=2)
        for r in results:
            self.assertTrue(r.name.startswith('idioms.assign.') or r.name.startswith('suites.polybench.cholesky.'), "Must only run filtered tests" )

        # Check report
        reports = list((self.rootdir /'results').glob('**/report.html'))
        self.assertEqual(len(reports),1)









if __name__ == '__main__':
    unittest.main()
