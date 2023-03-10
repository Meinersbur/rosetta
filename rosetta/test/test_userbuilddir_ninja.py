#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import tempfile
import unittest
import sys
import tempfile
import re
import io

from context import rosetta
import rosetta.util.invoke as invoke
import rosetta.runner
from rosetta.util.support import *
from rosetta.driver import *
from rosetta.util.support import *


class UserBuilddirNinja(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srcdir = mkpath(__file__).parent.parent.parent
        cls.test_dir = tempfile.TemporaryDirectory(prefix='userninja-')
        cls.builddir = mkpath(cls.test_dir)
        print("srcdir: ", cls.srcdir)
        print("builddir: ", cls.builddir)
        invoke.diag('cmake', '-S', cls.srcdir, '-B', cls. builddir,  '-GNinja', '-DCMAKE_BUILD_TYPE=Release',
                    '-DROSETTA_BENCH_FILTER=--filter=cholesky', cwd=cls.builddir, onerror=invoke.Invoke.EXCEPTION)

        cls.resultsdir = cls.builddir / 'results'
        cls.benchlistfile = cls.builddir / 'benchmarks' / 'benchlist.py'

    @classmethod
    def tearDownClass(cls):
        cls.test_dir.cleanup()

    def setUp(self):
        rosetta.runner.reset_registered_benchmarks()
        if self.resultsdir.exists():
            shutil.rmtree(self.resultsdir)
        # self.resultsdir.mkdir(parents=True)

    def test_build(self):
        rosetta.driver.driver_main(argv=[None, '--build'], mode=DriverMode.USERBUILDDIR,
                                   benchlistfile=self.benchlistfile, builddir=self.builddir, srcdir=self.srcdir)
        self.assertTrue((self.builddir / 'benchmarks' /
                        'suites.polybench.cholesky.serial').exists())

    def test_verify(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(Tee(f, sys.stdout)):
            rosetta.driver.driver_main(argv=[None, '--verify'], mode=DriverMode.USERBUILDDIR,
                                       benchlistfile=self.benchlistfile, builddir=self.builddir, srcdir=self.srcdir)

        s = f.getvalue()
        self.assertTrue(
            re.search(r'^Output of .* considered correct$', s, re.MULTILINE))
        self.assertFalse(re.search(r'^Array data mismatch\:', s, re.MULTILINE))

    def test_bench(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(Tee( f, sys.stdout)):
            rosetta.driver.driver_main(argv=[None, '--bench'], mode=DriverMode.USERBUILDDIR,
                                   benchlistfile=self.benchlistfile, builddir=self.builddir, srcdir=self.srcdir)

        # Check terminal output
        s=f.getvalue()
        self.assertTrue(re.search(r'Benchmark.+Wall', s, re.MULTILINE), "Evaluation table Header")
        self.assertTrue(re.search(r'suites\.polybench\.cholesky', s, re.MULTILINE), "Benchmark entry")

        # Check existance of the reportfile, only a single one is expected
        [reportfile] = list((self.builddir /'results').glob('*/report.html'))
        resultsubdir = reportfile.parent

        # Check benchmark results 
        results = list(resultsubdir.glob('*.xml'))
        self.assertTrue(len(results)>=1)
        for r in results:
            self.assertTrue(r.name.startswith('suites.polybench.cholesky.'), "Must only run filtered tests" )



if __name__ == '__main__':
    unittest.main()
