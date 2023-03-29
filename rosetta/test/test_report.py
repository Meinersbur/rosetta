#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from context import rosetta
import rosetta.driver
from rosetta.util.support import *

  
  
class ReportTests(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory(prefix='reporttest-')
        self.rootdir = mkpath( self.test_dir)
        print("rootdir: " , self.rootdir)

        self.report = None


    def tearDown(self):
        self.test_dir.cleanup()
        

    

    def createReport(self, reportfilename, resultsdir, args=[]):
        reportfile = self.rootdir / reportfilename
        rosetta.driver.driver_main(argv=[None, '--use-results-rdir', mkpath(__file__ ).parent / 'resultfiles'/ resultsdir, '--reportfile', reportfile] + args, mode= rosetta.driver.DriverMode.MANAGEDBUILDDIR)   
        with reportfile.open('r') as f:
            self.report = f.read()


    def test_single(self):
        self.createReport(reportfilename='singleresult.html', resultsdir = 'single')
        self.assertRegex(self.report, r'Benchmark\ Report')
        self.assertRegex(self.report, r'idioms\.assign')
        #self.assertRegex(self.report, r'serial') # TODO: Summarize common properties


    def test_multi_ppm(self):
        self.createReport(reportfilename='multiresult.html', resultsdir = 'multi_ppm'  )
        self.assertRegex(self.report, r'Benchmark\ Report')
        self.assertRegex(self.report, r'idioms\.assign')
        self.assertRegex(self.report, r'serial')
        self.assertRegex(self.report, r'cuda')


if __name__ == '__main__':
    unittest.main()
