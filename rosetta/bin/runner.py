#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import psutil
import subprocess
import pathlib
import os
import resource
import datetime
import json
import xml.etree.ElementTree as et


class BenchVariants:
    def __init__(self, default_size, serial=None, cuda=None):
        None



class BenchResult:
    def __init__(self,name:str, wtime : datetime.timedelta, rtime : datetime.timedelta,maxrss):
        self.name=name
        self.wtime=wtime
        self.rtime=rtime
        self.maxrss=maxrss



def run_gbench(exe):
    start = datetime.datetime.now()
    p = subprocess.Popen([exe],stdout=subprocess.PIPE,universal_newlines=True)
    #print([exe])

    

    stdout = p.stdout.read()
    unused_pid, exitcode, ru = os.wait4(p.pid, 0)
    stop = datetime.datetime.now()
    wtime = max(stop - start,datetime.timedelta(0))
    utime = ru.ru_utime
    stime = ru.ru_stime
    maxrss = ru.ru_maxrss

    benchmarks = et.fromstring(stdout)

    for benchmark in benchmarks:
        name = benchmark.attrib['name']
        n = benchmark.attrib['n']
        wallsum = 0
        usersum = 0
        kernelsum = 0
        count = len(benchmark)
        for it in benchmark:
            walltime = float(it.attrib['walltime'])
            wallsum += walltime
            usertime =float( it.attrib['usertime'])
            usersum += usertime
            kerneltime = float(it.attrib['kerneltime'])
            kernelsum += kerneltime
        yield BenchResult(name=name,wtime=walltime/count,rtime=usersum/count,maxrss=maxrss) 
        



def run_benchs(config:str=None,serial=[],cuda=[]):
    results = []
    for e in serial:
        results += list(run_gbench(exe=e))

    for e in cuda:
        results += list(run_gbench(exe=e))

    print("Name: WallTime RealTime MaxRSS")
    for r in results:
        print(f"{r.name}: {r.wtime} {r.rtime} {r.maxrss}")




def main(argv):
    parser = argparse.ArgumentParser(description="Benchmark runner", allow_abbrev=False)
    #parser.add_argument('--exe',    action='append', default=[], type=pathlib.Path,  help="Google Benchmark Executable")
    #parser.add_argument('--exedir', action='append', default=[], type=pathlib.Path, help="Google Benchmark Executable")
    #parser.add_argument('gbenchexe',nargs='+', help="Google Benchmark Executables")
    parser.add_argument('--serial', action='append', default=[], help="Google Benchmark Executables")
    parser.add_argument('--cuda', action='append', default=[], help="Google Benchmark Executables")
    args = parser.parse_args(argv[1:])




if __name__ == '__main__':
    retcode = main(argv=sys.argv)
    if retcode:
        exit(retcode)
