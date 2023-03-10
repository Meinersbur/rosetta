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
from .util.cmdtool import *
from .util.support import *
from .common import *
from .runner import Benchmark,do_run,get_problemsizefile,get_problemsize
from  . import  runner
from .evaluator import load_resultfiles
from . import runner




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
    for bench in runner. benchmarks:
        if filterfunc and not filterfunc(bench):
            continue
        ensure_reffile(bench, refdir=refdir, problemsizefile=problemsizefile)


def run_verify(problemsizefile, filterfunc=None, srcdir=None, refdir=None):
    problemsizefile = get_problemsizefile(
        srcdir=srcdir, problemsizefile=problemsizefile)

    #x = request_tempdir(prefix=f'verify')
    #tmpdir = mkpath(x.name)
    refdir.mkdir(exist_ok=True, parents=True)

    for e in runner.benchmarks:
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