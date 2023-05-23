#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import contextlib
import io
import re

from context import rosetta
import rosetta.driver
from rosetta.util.support import *


def tablecellre(*args):
    return '.+'.join(a.replace('\\', r'\\').replace('.', r'\.').replace('-', r'\-').replace('%', r'\%').replace(' ', r'\ ')  for a in args )


class EvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srcdir = mkpath(__file__ ).parent.parent.parent
        cls.test_dir = tempfile.TemporaryDirectory(prefix='evaltest-')
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

 
    def test_single(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(Tee( f, sys.stdout)):
            rosetta.driver.driver_main(argv=[None, '--evaluate', '--use-results-rdir', mkpath(__file__ ).parent / 'resultfiles'/ 'single' ], mode= rosetta.driver.DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir)     

        # Search for the table header
        s=f.getvalue().splitlines()
        while s:
            if re.search(tablecellre('Benchmark', 'timestamp', 'Wall', 'Max RSS'), s[0]):
                break
            s.pop(0)

        self.assertRegex(s[1], tablecellre('---',  '---') )
        self.assertRegex(s[2], tablecellre('idioms.assign',  '42.00', '162968') )
        self.assertRegex(s[3], tablecellre('idioms.assign',  '42.00', '147343') ) # Should it combine multiple different runs?


    def test_ppm(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(Tee( f, sys.stdout)):
            rosetta.driver.driver_main(argv=[None, '--evaluate', '--use-results-rdir', mkpath(__file__ ).parent / 'resultfiles'/ 'multi_ppm' ], mode= rosetta.driver.DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir)     

        # Search for the table header
        s=f.getvalue().splitlines()
        while s:
            if re.search(tablecellre('Benchmark', 'PPM', 'Wall'), s[0]):
                break
            s.pop(0)

        self.assertRegex(s[1], tablecellre('---', '---', '---'))
        self.assertRegex(s[2], tablecellre('idioms.assign', 'serial', '42.00' ))
        self.assertRegex(s[3], tablecellre('idioms.assign', 'cuda', '4.20' ))


    def test_ppm_compare(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(Tee( f, sys.stdout)):
            rosetta.driver.driver_main(argv=[None, '--evaluate', '--use-results-rdir', mkpath(__file__ ).parent / 'resultfiles'/ 'multi_ppm' , '--compare-by=ppm'], mode= rosetta.driver.DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir)     

        # Search for the table header
        s=f.getvalue().splitlines()
        while s:
            if re.search(tablecellre('Benchmark', 'Wall'), s[0]):
                break
            s.pop(0)

        self.assertRegex(s[1], tablecellre('serial', 'cuda'))
        self.assertRegex(s[2], tablecellre('---', '---', '---') )
        self.assertRegex(s[3], tablecellre('idioms.assign', '42.00', '4.20' ))


    def test_maxrss(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(Tee( f, sys.stdout)):
            rosetta.driver.driver_main(argv=[None, '--evaluate', '--use-results-rdir', mkpath(__file__ ).parent / 'resultfiles'/ 'multi_ppm' ], mode= rosetta.driver.DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir)     

        # Search for the table header
        s=f.getvalue().splitlines()
        while s:
            if re.search(tablecellre('Benchmark', 'PPM', 'Max RSS'), s[0]):
                break
            s.pop(0)

        self.assertRegex(s[1], tablecellre('---', '---', '---'))
        self.assertRegex(s[2], tablecellre('idioms.assign', 'serial', '162968' ))
        self.assertRegex(s[3], tablecellre('idioms.assign', 'cuda', '72788' ))


    def test_maxrss_compare(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(Tee( f, sys.stdout)):
            rosetta.driver.driver_main(argv=[None, '--evaluate', '--use-results-rdir', mkpath(__file__ ).parent / 'resultfiles'/ 'multi_ppm' , '--compare-by=ppm'], mode= rosetta.driver.DriverMode.MANAGEDBUILDDIR, rootdir=self.rootdir, srcdir=self.srcdir)     

        # Search for the table header
        s=f.getvalue().splitlines()
        while s:
            if re.search(tablecellre('Benchmark', 'Max RSS'), s[0]):
                break
            s.pop(0)

        self.assertRegex(s[1], tablecellre('serial', 'cuda'))
        self.assertRegex(s[2], tablecellre('---', '---', '---') )
        self.assertRegex(s[3], tablecellre('idioms.assign', '162968', '72788' ))
  



if __name__ == '__main__':
    unittest.main()
