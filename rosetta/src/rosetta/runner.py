#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import cwcwidth
import importlib.util
import importlib
import contextlib
import typing
import configparser
import io
from collections import defaultdict
import math
import colorama
import xml.etree.ElementTree as et
from typing import Iterable
import json
import datetime
import os
import pathlib
import subprocess
import argparse
import sys
from itertools import count
from cmath import exp
from .util.cmdtool import *
from .util.orderedset import OrderedSet
from .util.support import *
from .util import invoke
from .common import *
from .evaluator import subcommand_evaluate

# Not included batteries
# import tqdm # progress meter


class BenchVariants:
    def __init__(self, default_size, serial=None, cuda=None):
        None




def same_or_none(data):
    if not data:
        return None
    it = iter(data)
    common_value = None
    try:
        common_value = next(it)
        while True:
            next_value = next(it)
            if common_value != next_value:
                return None
    except StopIteration:
        return common_value







def do_run(bench, args, resultfile):
    exe = bench.exepath

    start = datetime.datetime.now()
    args.append(f'--xmlout={resultfile}')
    print("Executing", shjoin([exe] + args))
    #p = subprocess.Popen([exe] + args ,stdout=subprocess.PIPE,universal_newlines=True)
    p = subprocess.Popen([exe] + args)
    #stdout = p.stdout.read()
    # TODO: Integrate into invoke TODO: Fallback on windows TODO: should measure this in-process
    unused_pid, exitcode, ru = os.wait4(p.pid, 0)

    stop = datetime.datetime.now()
    p.wait()  # To let python now as well that it has finished

    assert resultfile.is_file(), "Expecting result file to be written by benchmark"

    wtime = max(stop - start, datetime.timedelta(0))
    utime = ru.ru_utime
    stime = ru.ru_stime
    maxrss = ru.ru_maxrss * 1024
    return resultfile










def run_gbench(bench, problemsizefile, resultfile):
    args = []
    if problemsizefile:
        args.append(f'--problemsizefile={problemsizefile}')
    return do_run(bench=bench, args=args, resultfile=resultfile)





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


def get_problemsizefile(srcdir=None, problemsizefile=None):
    if problemsizefile:
        if not problemsizefile.is_file():
            # TODO: Embed default sizes
            die(f"Problemsize file {problemsizefile} does not exist.", file=sys.stderr)
        return problemsizefile

    # Default, embedded into executable
    return None


def get_problemsize(bench: Benchmark, problemsizefile: pathlib.Path):
    if not problemsizefile:
        return bench.pbsize

    config = configparser.ConfigParser()
    config.read(problemsizefile)
    n = config.getint(bench.name, 'n')
    return n


def get_refpath(bench, refdir, problemsizefile):
    pbsize = get_problemsize(bench, problemsizefile=problemsizefile)
    reffilename = f"{bench.name}.{pbsize}.reference_output"
    refpath = refdir / reffilename
    return refpath


def ensure_reffile(bench: Benchmark, refdir, problemsizefile):
    refpath = get_refpath(bench, refdir=refdir, problemsizefile=problemsizefile)

    if refpath.exists():
        # Reference output already exists; check that it is the latest
        benchstat = bench.exepath.stat()
        refstat = refpath.stat()
        if benchstat.st_mtime < refstat.st_mtime:
            print(f"Reference output of {bench.name} already exists at {refpath} an is up-to-date")
            return
        print(f"Reference output {refpath} an is out-of-date")
        refpath.unlink()

    # Invoke reference executable and write to file
    args = [bench.exepath, f'--verify', f'--verifyfile={refpath}']
    if problemsizefile:
        args.append(f'--problemsizefile={problemsizefile}')
    invoke.call(*args, print_command=True)
    if not refpath.is_file():
        print(f"{refpath} not been written?")
        assert refpath.is_file()

    print(f"Reference output of {bench.name} written to {refpath}")


def ensure_reffiles(refdir, problemsizefile, filterfunc=None, srcdir=None):
    problemsizefile = get_problemsizefile(srcdir=srcdir, problemsizefile=problemsizefile)
    for bench in benchmarks:
        if filterfunc and not filterfunc(bench):
            continue
        ensure_reffile(bench, refdir=refdir, problemsizefile=problemsizefile)


# math.prod only available in Python 3.8
def prod(iter):
    result = 1
    for v in iter:
        result *= v
    return result


def run_verify(problemsizefile, filterfunc=None, srcdir=None, refdir=None):
    problemsizefile = get_problemsizefile(srcdir=srcdir, problemsizefile=problemsizefile)

    #x = request_tempdir(prefix=f'verify')
    #tmpdir = mkpath(x.name)
    refdir.mkdir(exist_ok=True, parents=True)

    for e in benchmarks:
        if filterfunc and not filterfunc(e):
            continue

        ensure_reffile(e, refdir=refdir, problemsizefile=problemsizefile)

        exepath = e.exepath
        refpath = get_refpath(e, refdir=refdir, problemsizefile=problemsizefile)
        pbsize = get_problemsize(e, problemsizefile=problemsizefile)

        testoutpath = request_tempfilename(subdir='verify', prefix=f'{e.name}_{e.ppm}_{pbsize}', suffix='.testout')
        # tmpdir / f'{e.name}_{e.ppm}_{pbsize}.testout'

        args = [exepath, f'--verify', f'--verifyfile={testoutpath}']
        if problemsizefile:
            args.append(f'--problemsizefile={problemsizefile}')
        p = invoke.call(*args, return_stdout=True, print_command=True)

        with refpath.open() as fref, testoutpath.open() as ftest:
            while True:
                refline = fref.readline()
                testline = ftest.readline()

                # Reached end-of-file?
                if not refline and not testline:
                    break

                refspec, refdata = refline.split(':', maxsplit=1)
                refspec = refspec.split()
                refkind = refspec[0]
                refformat = refspec[1]
                refdim = int(refspec[2])
                refshape = [int(i) for i in refspec[3:3 + refdim]]
                refname = refspec[3 + refdim] if len(refspec) > 3 + refdim else None
                refcount = prod(refshape)

                refdata = [float(v) for v in refdata.split()]
                if refcount != len(refdata):
                    die(f"Unexpected array items in {refname}: {refcount} vs {len(refdata)}")

                testspec, testdata = testline.split(':', maxsplit=1)
                testspec = testspec.split()
                testkind = testspec[0]
                testformat = testspec[1]
                testdim = int(testspec[2])
                testshape = [int(i) for i in testspec[3:3 + testdim]]
                testname = testspec[3 + testdim] if len(testspec) > 3 + testdim else None
                testcount = prod(testshape)

                testdata = [float(v) for v in testdata.split()]
                if testcount != len(testdata):
                    die(f"Unexpected array items in {testname}: {testcount} vs {len(testdata)}")

                if refname is not None and testname is not None and refname != testname:
                    die(f"Array names {refname} and {testname} disagree")

                for i, (refv, testv) in enumerate(zip(refdata, testdata)):
                    coord = [str((i // prod(refshape[0:j])) % refshape[j]) for j in range(0, refdim)]
                    coord = '[' + ']['.join(coord) + ']'

                    if math.isnan(refv) and math.isnan(testv):
                        print(f"WARNING: NaN in both outputs at {refname}{coord}")
                        continue
                    if math.isnan(refv):
                        die(f"Array data mismatch: Ref contains NaN at {refname}{coord}")
                    if math.isnan(testv):
                        die(f"Array data mismatch: Output contains NaN at {testname}{coord}")

                    mid = (abs(refv) + abs(testv)) / 2
                    absd = abs(refv - testv)
                    if mid == 0:
                        reld = 0 if absd == 0 else math.inf
                    else:
                        reld = absd / mid
                    if reld > 1e-4:  # TODO: Don't hardcode difference
                        print(f"While comparing {refpath} and {testoutpath}:")
                        die(f"Array data mismatch: {refname}{coord} = {refv} != {testv} = {testname}{coord} (Delta: {absd}  Relative: {reld})")

        print(f"Output of {e.exepath} considered correct")


def make_resultssubdir(within=None):
    global resultsdir
    within = within or resultsdir
    assert within
    now = datetime.datetime.now()
    i = 0
    suffix = ''
    while True:
        resultssubdir = within / f"{now:%Y%m%d_%H%M}{suffix}"
        if not resultssubdir.exists():
            resultssubdir.mkdir(parents=True)
            return resultssubdir
        i += 1
        suffix = f'_{i}'


def run_bench(problemsizefile=None, srcdir=None, resultdir=None):
    problemsizefile = get_problemsizefile(srcdir, problemsizefile)

    results = []
    resultssubdir = make_resultssubdir(within=resultdir)
    for e in benchmarks:
        thisresultdir = resultssubdir
        configname = e.configname
        if configname:
            thisresultdir /= configname
        thisresultdir /= f'{e.name}.{e.ppm}.xml'
        results .append(run_gbench(e, problemsizefile=problemsizefile, resultfile=thisresultdir))
    return results


def custom_bisect_left(lb, ub, func):
    assert ub >= lb
    while True:
        if lb == ub:
            return lb
        mid = (lb + ub + 1) // 2
        result = func(mid)
        if result < 0:
            # Go smaller
            ub = mid - 1
            continue
        if result > 0:
            # Go larger, keep candidate as possible result
            lb = mid
            continue
        # exact match?
        return mid


mytempdir = None
globalctxmgr = contextlib. ExitStack()


def request_tempdir(subdir=None):
    global mytempdir
    if mytempdir:
        return mytempdir
    x = tempfile.TemporaryDirectory(prefix=f'rosetta-')  # TODO: Option to not delete / keep in current directory
    mytempdir = mkpath(globalctxmgr.enter_context(x))
    return mytempdir


def request_tempfilename(prefix=None, suffix=None, subdir=None):
    tmpdir = request_tempdir(subdir=subdir)
    candidate = tmpdir / f'{prefix}{suffix}'
    i = 0
    while candidate.exists():
        candidate = tmpdir / f'{prefix}-{i}{suffix}'
        i += 1

    return candidate

# TODO: merge with run_gbench
# TODO: repeats for stability


def probe_bench(bench: Benchmark, limit_walltime, limit_rss, limit_alloc):
    assert limit_walltime or limit_rss or limit_alloc, "at least one limit required"

    def is_too_large(result):
        if limit_walltime is not None and result.durations['walltime'].mean >= limit_walltime:
            return True
        if limit_rss is not None and result.maxrss >= limit_rss:
            return True
        if limit_alloc is not None and result.peakalloc >= limit_alloc:
            return True
        return False

    # Find a rough ballpark
    lower_n = 1
    n = 1

    # Bisect between lower_n and n

    def func(n):
        resultfile = request_tempfilename(subdir='probe', prefix=f'{bench.target}-pbsize{n}', suffix='.xml')
        do_run(bench, args=[f'--pbsize={n}', '--repeats=1'], resultfile=resultfile)
        [result] = load_resultfiles([resultfile])
        if is_too_large(result):
            return -1
        return 1

    while func(n) != -1:
        lower_n = n
        n *= 2

    return custom_bisect_left(lower_n, n - 1, func)


def run_probe(problemsizefile, limit_walltime, limit_rss, limit_alloc):
    if not problemsizefile:
        die("Problemsizes required")

    problemsizecontent = []
    for bench in benchmarks:
        n = probe_bench(bench=bench, limit_walltime=limit_walltime, limit_rss=limit_rss, limit_alloc=limit_alloc)

        problemsizecontent.extend(
            [f"[{bench.name}]",
             f"n={n}",
             ""
             ]
        )
    with problemsizefile.open(mode='w+') as f:
        for line in problemsizecontent:
            print(line, file=f)


def runner_main(builddir):
    runner_main_run()


def runner_main_run(srcdir, builddir):
    with globalctxmgr:
        parser = argparse.ArgumentParser(description="Benchmark runner", allow_abbrev=False)
        add_boolean_argument(parser, 'buildondemand', default=True, help="build to ensure executables are up-to-data")
        resultdir = builddir / 'results'
        subcommand_run(parser, None, srcdir, builddirs=[builddir], refbuilddir=builddir, resultdir=resultdir)
        subcommand_evaluate(parser,None,resultfiles=None)
        args = parser.parse_args(sys.argv[1:])

        resultfiles=  subcommand_run(None, args, srcdir, builddirs=[
                       builddir], buildondemand=args.buildondemand, refbuilddir=builddir, resultdir=resultdir)
        subcommand_evaluate(None,args,resultfiles)




def subcommand_run(parser, args, srcdir, buildondemand: bool = False, builddirs=None,
                   refbuilddir=None, filterfunc=None, resultdir=None):
    """
The common functionality of the probe/verify/bench per-builddir scripts and the benchmark.py multi-builddir driver.

General pipeline:
1. configure (handled by benchmark.py; per-builddir setup is pre-configured)
2. build
3. probe (reconfigure/rebuild required after probing)
4. verify
5. bench
6. evaluate

Parameters
----------
parser : ArgumentParser
    ArgumentParser from argparse for adding arguments
args
    Parsed command line from argparse
srcdir
    Root of the source repository (where the benchmarks and rosetta folders are in);
    FIXME: Should be the benchmarks folder
buildondemand
    False: Called from make/ninja, targets already built
    True: Called from script, executables may need to be rebuilt
builddirs:
    per-builddir: The root of CMake's binary dir
    multi-builddor: List of builddires that have been configured
refbuilddir:
    CMake binary root dir that creates the reference outputs, but is itself not benchmarked
filterfunc:
    Run/evaluate only those benchmarks for which this function returns true
resultdir:
    Where to put the benchmark results xml files.
"""
    if parser:
        parser.add_argument('--problemsizefile', type=pathlib.Path, help="Problem sizes to use (.ini file)")
        parser.add_argument('--verbose', '-v', action='count')

        # Command
        add_boolean_argument(parser, 'probe', default=False, help="Enable probing")
        parser.add_argument('--limit-walltime', type=parse_time)
        parser.add_argument('--limit-rss', type=parse_memsize)
        parser.add_argument('--limit-alloc', type=parse_memsize)

        # Verify step
        add_boolean_argument(parser, 'verify', default=False, help="Enable check step")

        # Run step
        add_boolean_argument(parser, 'bench', default=None, help="Enable run step")


    if args:
        # If neither no action is specified, enable --bench implicitly unless --no-bench
        probe = args.probe
        verify = args.verify
        bench = args.bench
        eval = args.evaluate
        if bench is None and not verify and not probe:
            bench = True

        if probe:
            assert args.problemsizefile, "Requires to set a problemsizefile to set"
            run_probe(problemsizefile=args.problemsizefile, limit_walltime=args.limit_walltime,
                      limit_rss=args.limit_rss, limit_alloc=args.limit_alloc)

        if verify:
            refdir = refbuilddir / 'refout'
            run_verify(problemsizefile=args.problemsizefile, refdir=refdir)

        resultfiles=None
        if bench:
            resultfiles = run_bench(srcdir=srcdir, problemsizefile=args.problemsizefile, resultdir=resultdir)

        return resultfiles




# TODO: Integrate into subcommand_run
def runner_main_verify(builddir, srcdir):
    parser = argparse.ArgumentParser(description="Benchmark verification", allow_abbrev=False)
    parser.add_argument('--problemsizefile', type=pathlib.Path, help="Problem sizes to use (.ini file)")
    add_boolean_argument(parser, 'buildondemand', default=True)  # TODO: implement

    args = parser.parse_args()

    refdir = builddir / 'refout'
    return run_verify(problemsizefile=args.problemsizefile, refdir=refdir)


def runner_main_probe(builddir):
    die("Not yet implemented")


resultsdir = None


def rosetta_config(resultsdir):
    def set_global(dir):
        global resultsdir
        resultsdir = mkpath(dir)
    # TODO: Check if already set and different
    set_global(resultsdir)


benchlistfile = None
import_is_ref = None
benchmarks: typing .List[Benchmark] = []


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
        spec = importlib.util.spec_from_file_location(filename.stem, str(filename))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        benchlistfile = None
        import_is_ref = None


def gen_reference(exepath, refpath, problemsizefile):
    args = [exepath, f'--verify', f'--problemsizefile={problemsizefile}']
    invoke.call(*args, stdout=refpath, print_stderr=True, print_command=True)


def main(argv):
    colorama.init()
    parser = argparse.ArgumentParser(description="Benchmark runner", allow_abbrev=False)
    parser.add_argument('--gen-reference', nargs=2, type=pathlib.Path, help="Write reference output file")
    parser.add_argument('--problemsizefile', type=pathlib.Path)
    args = parser.parse_args(argv[1:])

    if args.gen_reference:
        gen_reference(*args.gen_reference, problemsizefile=args.problemsizefile)



if __name__ == '__main__':
    retcode = main(argv=sys.argv)
    if retcode:
        exit(retcode)
