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
from .runner import Benchmark,benchmarks,get_problemsizefile,make_resultssubdir,do_run
from  . import  runner
from .evaluator import load_resultfiles


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
    n = 2

    # Bisect between lower_n and n

    def func(n):
        resultfile = request_tempfilename(
            subdir='probe', prefix=f'{bench.target}-pbsize{n}', suffix='.xml')
        do_run(bench, args=[f'--pbsize={n}',
               '--repeats=1'], resultfile=resultfile)
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
    for bench in runner.benchmarks:
        n = probe_bench(bench=bench, limit_walltime=limit_walltime,
                        limit_rss=limit_rss, limit_alloc=limit_alloc)

        problemsizecontent.extend(
            [f"[{bench.name}]",
             f"n={n}",
             ""
             ]
        )
    with problemsizefile.open(mode='w+') as f:
        for line in problemsizecontent:
            print(line, file=f)

