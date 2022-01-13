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


class BenchResult:
    def __init__(self,name:str,wtime : datetime.timedelta,rtime : datetime.timedelta,maxrss):
        self.name=name
        self.wtime=wtime
        self.rtime=rtime
        self.maxrss=maxrss



def run_gbench(exe):
    start = datetime.datetime.now()
    p = subprocess.Popen([exe, '--benchmark_format=json'],stdout=subprocess.PIPE,text=True)
    
    unused_pid, exitcode, ru = os.wait4(p.pid, 0)
    stop = datetime.datetime.now()
    wtime = max(stop - start,datetime.timedelta(0))
    utime = ru.ru_utime
    stime = ru.ru_stime
    maxrss = ru.ru_maxrss
    stdout = p.stdout.read()
    data = json.loads(stdout)

    name = data['benchmarks'][0]['name']
    reps = data['benchmarks'][0]['repetitions']
    iters = data['benchmarks'][0]['iterations']
    ctime = data['benchmarks'][0]['cpu_time']
    rtime = data['benchmarks'][0]['real_time']
    tunit = data['benchmarks'][0]['time_unit']

    benchmarks = data['benchmarks']
    assert len(benchmarks)==1,"For accurate results, just one benchmark per executable"

    #print(f"maxrss={ru.ru_maxrss}, stdout={data['benchmarks'][0]}")
    for d in data['benchmarks']:
        yield BenchResult(name=d['name'],wtime=wtime,rtime=datetime.timedelta(milliseconds=d['real_time']),maxrss=maxrss)


def main(argv):
    parser = argparse.ArgumentParser(description="Benchmark runner", allow_abbrev=False)
    parser.add_argument('--exe',    action='append', default=[], type=pathlib.Path,  help="Google Benchmark Executable")
    parser.add_argument('--exedir', action='append', default=[], type=pathlib.Path, help="Google Benchmark Executable")
    args = parser.parse_args(argv[1:])

    results = []
    for e in args.exe:
        results += list(run_gbench(exe=e))
    for p in args.exedir:
        for e in p.glob("**/*.seq"):
            results += list(run_gbench(exe=e))

    print("Name: WallTime RealTime MaxRSS")
    for r in results:
        print(f"{r.name}: {r.wtime} {r.rtime} {r.maxrss}")


if __name__ == '__main__':
    if retcode := main(argv=sys.argv):
        exit(retcode)


