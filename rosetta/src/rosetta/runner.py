# -*- coding: utf-8 -*-

"""Run the Benchmarks"""

import configparser
import datetime
import os
import pathlib
import subprocess
import sys

from .util.cmdtool import *
from .util.support import *
from .util import invoke
from .common import *
from .registry import Benchmark
from . import registry
from .filtering import *


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


def do_run(bench, args, resultfile, timestamp=None):
    exe = bench.exepath

    start = datetime.datetime.now()
    args.append(f'--xmlout={resultfile}')
    if timestamp:
        args.append(f'--timestamp={timestamp.isoformat(sep=" ")}')
    # print("Executing", shjoin([exe] + args))

    invoke.diag(exe, *args,
                setenv={'OMP_TARGET_OFFLOAD': 'mandatory'}  # TODO: Make configurable per-PPM
                )
    assert resultfile.is_file(), "Expecting result file to be written by benchmark"
    return resultfile

    # p = subprocess.Popen([exe] + args ,stdout=subprocess.PIPE,universal_newlines=True)
    p = subprocess.Popen([exe] + args)
    # stdout = p.stdout.read()
    # TODO: Integrate into invoke TODO: Fallback on windows TODO: should measure this in-process
    unused_pid, exitcode, ru = os.wait4(p.pid, 0)

    stop = datetime.datetime.now()
    p.wait()  # To let python now as well that it has finished

    wtime = max(stop - start, datetime.timedelta(0))
    utime = ru.ru_utime
    stime = ru.ru_stime
    maxrss = ru.ru_maxrss * 1024
    return resultfile


def run_gbench(bench, problemsizefile, resultfile, timestamp):
    args = []
    if problemsizefile:
        args.append(f'--problemsizefile={problemsizefile}')
    return do_run(bench=bench, args=args, resultfile=resultfile, timestamp=timestamp)


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


def make_resultssubdir(within):
    # global resultsdir
    # within = within or resultsdir
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


def run_bench(problemsizefile=None, srcdir=None, resultdir=None, args=None):
    problemsizefile = get_problemsizefile(srcdir, problemsizefile)
    results = []
    resultssubdir = make_resultssubdir(within=resultdir)
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    filtered_benchmarks = get_filtered_benchmarks(registry.benchmarks, args)
    for e in filtered_benchmarks:
        thisresultdir = resultssubdir
        configname = e.configname
        if configname:
            thisresultdir /= configname
        thisresultdir /= f'{e.name}.{e.ppm}.xml'
        results.append(run_gbench(
            e, problemsizefile=problemsizefile, resultfile=thisresultdir, timestamp=timestamp))
    return results, resultssubdir


if __name__ == '__main__':
    retcode = main(argv=sys.argv)
    if retcode:
        exit(retcode)
