#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from context import rosetta
import rosetta.runner
import tempfile
import  rosetta.util.invoke as invoke
from rosetta.util.support import *
import unittest
import sys
import tempfile
from  rosetta.driver import *
from rosetta.util.support import *
import shutil




class UserBuilddirMulticonfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srcdir = mkpath(__file__ ).parent.parent.parent
        cls.test_dir = tempfile.TemporaryDirectory(prefix='userninja-')
        cls.builddir = mkpath(cls.test_dir)
        print("srcdir: " , cls.srcdir)
        print("builddir: " , cls.builddir)
        invoke.diag('cmake', '-S', cls.srcdir, '-B',cls. builddir,  '-GNinja Multi-Config',  '-DROSETTA_BENCH_FILTER=--filter=cholesky', cwd= cls.builddir,onerror=invoke.Invoke.EXCEPTION)

        cls.resultsdir= cls.builddir / 'results'
        cls.benchlistfile = cls.builddir / 'benchmarks' / 'benchlist.py'


    @classmethod
    def tearDownClass(cls):
        cls.test_dir.cleanup()


    def setUp(self):
        rosetta.runner.reset_registered_benchmarks()
        if self.resultsdir.exists():
            shutil.rmtree(self.resultsdir)
        #self.resultsdir.mkdir(parents=True)


    def test_build(self):
        rosetta.driver.driver_main(  argv=[None, '--build'], mode=DriverMode.USERBUILDDIR, benchlistfile=  self.benchlistfile, builddir=self.builddir, srcdir=self.srcdir  )     
        self.assertTrue((self.builddir / 'benchmarks' / 'Release' / 'suites.polybench.cholesky.serial').exists())


    def test_verify(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(Tee( f, sys.stdout)):
            rosetta.driver.driver_main(  argv=[None, '--verify'], mode=DriverMode.USERBUILDDIR, benchlistfile=self.benchlistfile, builddir=self.builddir, srcdir=self.srcdir  )       

        s = f.getvalue()
        self.assertTrue(re.search(r'^Output of .*\.cholesky\..* considered correct$',s, re.MULTILINE ))
        self.assertFalse(re.search(r'^Array data mismatch\:',s, re.MULTILINE));
        

    def test_bench(self):
        rosetta.driver.driver_main( argv=[None, '--bench'], mode=DriverMode.USERBUILDDIR,benchlistfile=  self.benchlistfile, builddir=self.builddir, srcdir=self.srcdir)    

        # Check benchmark results 
        results = list((self.builddir /'results').glob('**/*.xml'))
        self.assertTrue(len(results)>=1)
        for r in results:
            self.assertTrue(r.name.startswith('suites.polybench.cholesky.'), "Must only run filtered tests" )

        # Check report
        reports = list((self.builddir /'results').glob('report_*.html'))
        self.assertEqual(len(reports),1)





if __name__ == '__main__':
    unittest.main()


