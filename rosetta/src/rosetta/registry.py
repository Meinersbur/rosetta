# -*- coding: utf-8 -*-

"""Global registry of benchmarks"""

import typing

from .util.cmdtool import *
from .util.support import *
from .common import *





runtime = NamedSentinel('runtime')
compiletime = NamedSentinel('compiletime')

class Param:
    def __init__(self,name,min=None,max=None, verify=None,train=None,ref=None,  choices=None,evaltime=None,*args):
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




class SizeParam(GenParam):
    """Parameter to generate benchmarks that differ in the working set size; Its value is probed to be as large as possible without violating constraints
    
    One one size parameter allowed: 'n'
    """
    def __init__(self,*args,**kwargs):
         super().__init__(*args,**kwargs)

class RealtypeParam(GenParam):
    """For selecting the floating point precision"""



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



class ComparableBenchmark:
    """Set of benchmarks that are expected to compute the same result, PPM and TuneParam may very, but not GenParams or SizeParam"""
    def __init__(self,basename,params=[]):
        self.benchmarks = []
        self.basename = basename
        assert all( not issubclass(p,TuneParam) for p in params )
        self.params = params


    def add(self,benchmark):
        self.benchmarks.append(benchmark)

    @property 
    def reference(self):
        """Return the executable that computes the reference output"""
        for b in self.benchmarks:
            if b.ppm == 'serial': # TODO: Make configurable
                return b
        log.warn(f"No reference found for {self.basename}; using {self.benchmarks[0]}")
        return self.benchmarks[0]



# TOOD: Rename: TunedBenchmark(?)
# TOOD: Fixed static parameters?
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

        # The set of comparable benchmarks   it belongs to 
        self.comparable = None


    @property
    def name(self):
        return self.basename

 

class TunedBenchmark: 
    """Benchmark with fixed static and dynamic parameters"""
    def __init__(self,executable:Benchmark):
        self.executable = executable


benchlistfile = None
import_is_ref = None
benchmarks: typing.List[Benchmark] = []

comparables = []



def register_benchmark(basename, target, exepath, buildtype, ppm, configname,
                       benchpropfile=None, compiler=None, compilerflags=None, pbsize=None):
    assert basename is not None

    bench = Benchmark(basename=basename, target=target, exepath=mkpath(exepath), buildtype=buildtype, ppm=ppm, configname=configname,
                      benchpropfile=benchpropfile, compiler=compiler, compilerflags=compilerflags, pbsize=pbsize, benchlistfile=benchlistfile, is_ref=import_is_ref)
    benchmarks.append(bench)
    global comparables
    for c in comparables:
        if c.basename != basename:
            continue
        # TODO: Compare GenParams are the same
        comparable = c
        break
    else:
        comparable  = ComparableBenchmark(basename=basename)
        comparables.append(comparable)
    bench.comparable = comparable
    comparable.add(bench)



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
