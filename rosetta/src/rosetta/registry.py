# -*- coding: utf-8 -*-

"""Global registry of benchmarks"""

import typing
import configparser
import math
import colorama
import datetime
import os
import pathlib
import subprocess
import sys

from .util.cmdtool import *
from .util.support import *
from .util import invoke
from .common import *









class Benchmark:
    def __init__(self, basename, target, exepath, buildtype, ppm, configname, sources=None,
                 benchpropfile=None, compiler=None, compilerflags=None, pbsize=None, benchlistfile=None, is_ref=None):
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
