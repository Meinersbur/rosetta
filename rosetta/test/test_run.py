#! /usr/bin/env python3
# -*- coding: utf-8 -*-


from context import rosetta
import rosetta.scripts.run
import unittest
import tempfile
import  rosetta.util.invoke as invoke
import pathlib
from rosetta.util.support import *
import rosetta.runner

srcdir = mkpath(__file__ ).parent.parent.parent

class RunTests(unittest.TestCase):
    def test_userbuilddir_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = mkpath(tmpdir)
            invoke.diag('cmake', '-S', srcdir, '-B', tmpdir,  '-GNinja',  '-DCMAKE_BUILD_TYPE=Release', cwd=tmpdir)
            invoke.diag('cmake', '--build' , tmpdir,  '--target', 'suites.polybench.cholesky.serial', cwd=tmpdir)
            rosetta.runner.register_benchmark(basename='suites.polybench.cholesky', ppm='serial', target='suites.polybench.cholesky.serial', exepath=tmpdir /  'benchmarks' / 'suites.polybench.cholesky.serial', buildtype='Release', configname='', benchpropfile='/home/meinersbur/build/rosetta/release/benchmarks/suites.polybench.cholesky.serial.Release.benchprop.cxx', compiler='/usr/bin/c++', compilerflags='-fmax-errors=1 -O3 -DNDEBUG', pbsize=200)
            rosetta.runner.runner_main_run_argv(argv=[], srcdir=srcdir,builddir=tmpdir)
            #rosetta.scripts.run.main(argv=[], rootdir=tmpdir )





if __name__ == '__main__':
    unittest.main()


