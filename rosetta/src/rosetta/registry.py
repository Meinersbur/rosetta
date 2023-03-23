# -*- coding: utf-8 -*-

"""Global registry of benchmarks"""

import typing

from .util.cmdtool import *
from .util.support import *
from .common import *





runtime = NamedSentinel('runtime')
compiletime = NamedSentinel('compiletime')

class Param:
    def __init__(self,name,choices=None,evaltime=None,*args):
        self.name = name
        self.choices=choices
        self.allow_compiletime = None
        self.allow_runtime = None

        for a in itertools.chain(  ensure_list(evaltime),args):
            if a == compiletime :
                self.allow_compiletime = True
            elif a == compiletime:
                self.allow_runtime=True
            else:
                raise Exception(f"Unexpected parameter {a}")

        if self.allow_compiletime is None and self.allow_runtime is None:
             self.allow_compiletime =True
             self.allow_runtime=True

        self.allow_compiletime = first_defined(self.allow_compiletime,False)
        self.allow_runtime = first_defined(self.allow_runtime,False)



class GenParam(Param):
    """Parameter used to generate benchmarks (e.g. real=float or double); All possible combinations are selected"""
    def __init__(self,*args,**kwargs):
         super().__init__(*args,**kwargs)


class SizeParam(Param):
    """Parameter to generate benchmarks that differ in the working set size; Its value is probed to be as large as possible without violating constraints
    
    One one size parameter allowed: 'n'
    """
    def __init__(self,*args,**kwargs):
         super().__init__(*args,**kwargs)



class TuneParam(Param):
    """Parameter that does not have an influence on the output; Its value is tuned to optimize a criterium (usually minimize execution time) under constraints"""
    def __init__(self,*args,**kwargs):
         super().__init__(*args,**kwargs)




class UnsizedBenchmark:
    """Fixed generator parameters (including ppm)"""
    pass


class SizedBenchmark:
    """Fixed generator (and size, if compiletime) parameters"""
    pass


# TOOD: Rename: TunedBenchmark
class Benchmark:
    """A benchmark executable with fixed static parameters"""
    def __init__(self, basename, target, exepath, buildtype, ppm, configname, sources=None,
                 benchpropfile=None, compiler=None, compilerflags=None, pbsize=None, benchlistfile=None, is_ref=None, params=[]):
        self.basename = basename
        self.target = target
        self.exepath = exepath
        self.buildtype = buildtype
        self.ppm = ppm
        self.configname = configname
        self.sources = [mkpath(s) for s in sources] if sources else None
        self.benchpropfile = benchpropfile
        self.compiler = mkpath(compiler)
        self.compilerflags = compilerflags
        self.pbsize = pbsize  # default problemsize
        self.benchlistfile = benchlistfile
        self.is_ref = is_ref

    @property
    def name(self):
        return self.basename

 

benchlistfile = None
import_is_ref = None
benchmarks: typing.List[Benchmark] = []


def register_benchmark(basename, target, exepath, buildtype, ppm, configname,
                       benchpropfile=None, compiler=None, compilerflags=None, pbsize=None):
    bench = Benchmark(basename=basename, target=target, exepath=mkpath(exepath), buildtype=buildtype, ppm=ppm, configname=configname,
                      benchpropfile=benchpropfile, compiler=compiler, compilerflags=compilerflags, pbsize=pbsize, benchlistfile=benchlistfile, is_ref=import_is_ref)
    benchmarks.append(bench)


def load_register_file(filename, is_ref=False):
    global benchlistfile, import_is_ref
    import importlib

    filename = mkpath(filename)
    benchlistfile = filename
    import_is_ref = is_ref
    try:
        spec = importlib.util.spec_from_file_location(
            filename.stem, str(filename))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        benchlistfile = None
        import_is_ref = None

# TODO: Use global contenxt manager from support
def reset_registered_benchmarks():
    """
Reset loaded benchmarks for unittsts

A better approach would be if load_register_file returns the availailable benchmarks and the caller to pass them on
    """
    global benchmarks
    benchmarks = []
