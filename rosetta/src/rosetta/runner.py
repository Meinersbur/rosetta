# -*- coding: utf-8 -*-

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
            die(f"Problemsize file {problemsizefile} does not exist.",
                file=sys.stderr)
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
    refpath = get_refpath(bench, refdir=refdir,
                          problemsizefile=problemsizefile)

    if refpath.exists():
        # Reference output already exists; check that it is the latest
        benchstat = bench.exepath.stat()
        refstat = refpath.stat()
        if benchstat.st_mtime < refstat.st_mtime:
            print(
                f"Reference output of {bench.name} already exists at {refpath} an is up-to-date")
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
    problemsizefile = get_problemsizefile(
        srcdir=srcdir, problemsizefile=problemsizefile)
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
    problemsizefile = get_problemsizefile(
        srcdir=srcdir, problemsizefile=problemsizefile)

    #x = request_tempdir(prefix=f'verify')
    #tmpdir = mkpath(x.name)
    refdir.mkdir(exist_ok=True, parents=True)

    for e in benchmarks:
        if filterfunc and not filterfunc(e):
            continue

        ensure_reffile(e, refdir=refdir, problemsizefile=problemsizefile)

        exepath = e.exepath
        refpath = get_refpath(
            e, refdir=refdir, problemsizefile=problemsizefile)
        pbsize = get_problemsize(e, problemsizefile=problemsizefile)

        testoutpath = request_tempfilename(
            subdir='verify', prefix=f'{e.name}_{e.ppm}_{pbsize}', suffix='.testout')
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
                refname = refspec[3 +
                                  refdim] if len(refspec) > 3 + refdim else None
                refcount = prod(refshape)

                refdata = [float(v) for v in refdata.split()]
                if refcount != len(refdata):
                    die(
                        f"Unexpected array items in {refname}: {refcount} vs {len(refdata)}")

                testspec, testdata = testline.split(':', maxsplit=1)
                testspec = testspec.split()
                testkind = testspec[0]
                testformat = testspec[1]
                testdim = int(testspec[2])
                testshape = [int(i) for i in testspec[3:3 + testdim]]
                testname = testspec[3 +
                                    testdim] if len(testspec) > 3 + testdim else None
                testcount = prod(testshape)

                testdata = [float(v) for v in testdata.split()]
                if testcount != len(testdata):
                    die(
                        f"Unexpected array items in {testname}: {testcount} vs {len(testdata)}")

                if refname is not None and testname is not None and refname != testname:
                    die(f"Array names {refname} and {testname} disagree")

                for i, (refv, testv) in enumerate(zip(refdata, testdata)):
                    coord = [str((i // prod(refshape[0:j])) % refshape[j])
                             for j in range(0, refdim)]
                    coord = '[' + ']['.join(coord) + ']'

                    if math.isnan(refv) and math.isnan(testv):
                        print(
                            f"WARNING: NaN in both outputs at {refname}{coord}")
                        continue
                    if math.isnan(refv):
                        die(
                            f"Array data mismatch: Ref contains NaN at {refname}{coord}")
                    if math.isnan(testv):
                        die(
                            f"Array data mismatch: Output contains NaN at {testname}{coord}")

                    mid = (abs(refv) + abs(testv)) / 2
                    absd = abs(refv - testv)
                    if mid == 0:
                        reld = 0 if absd == 0 else math.inf
                    else:
                        reld = absd / mid
                    if reld > 1e-4:  # TODO: Don't hardcode difference
                        print(f"While comparing {refpath} and {testoutpath}:")
                        die(
                            f"Array data mismatch: {refname}{coord} = {refv} != {testv} = {testname}{coord} (Delta: {absd}  Relative: {reld})")

        print(f"Output of {e.exepath} considered correct")


def make_resultssubdir(within):
    #global resultsdir
    #within = within or resultsdir
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
        results .append(run_gbench(
            e, problemsizefile=problemsizefile, resultfile=thisresultdir))
    return results,resultssubdir


def custom_bisect_left(lb, ub, func):
    assert ub >= lb
    while True:
        if lb == ub:
            return lb
        mid = (lb + ub + 1) // 2
        result = func(mid)
        if result < 0:
            # Go smaller
            assert ub > mid - 1, "Require the bisect range to become smaller"
            ub = mid - 1
            continue
        if result > 0:
            # Go larger, keep candidate as possible result
            assert lb < mid , "Require the bisect range to become smaller"
            lb = mid
            continue
        # exact match?
        return mid



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


def reset_registered_benchmarks():
    """
Reset loaded benchmarks for unittsts

A better approach would be if load_register_file returns the availailable benchmarks and the caller to pass them on
    """
    global benchmarks
    benchmarks = []


def gen_reference(exepath, refpath, problemsizefile):
    args = [exepath, f'--verify', f'--problemsizefile={problemsizefile}']
    invoke.call(*args, stdout=refpath, print_stderr=True, print_command=True)


def main(argv):
    colorama.init()
    parser = argparse.ArgumentParser(
        description="Benchmark runner", allow_abbrev=False)
    parser.add_argument('--gen-reference', nargs=2,
                        type=pathlib.Path, help="Write reference output file")
    parser.add_argument('--problemsizefile', type=pathlib.Path)
    args = parser.parse_args(argv[1:])

    if args.gen_reference:
        gen_reference(*args.gen_reference,
                      problemsizefile=args.problemsizefile)


if __name__ == '__main__':
    retcode = main(argv=sys.argv)
    if retcode:
        exit(retcode)
