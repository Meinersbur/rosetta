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
import termcolor 



class Table :
    def __init__(self):
        self.columns = []
        self.column_titles = dict()
        self.column_formatters = dict()
        self.rows = []

    def add_column(self, name, title=None, formatter=None):
        self.columns.append(name)
        if title is not None:
            self.column_titles[name] = title
        if formatter is not None:
            self.column_formatters[name] = formatter

    def add_row(self, **kwargs):
        self.rows .append(kwargs)

    def print(self):
        # Content table
        #matrix = [[r.get(k) for r in self .rows] for k in self.columns] 
        matrix=[]
        nrows = len(self.rows)
        ncols = len(self.columns)

        collen = []
        titles = []
        for i,name in enumerate(self,self.columns):
            vals = [r.get(name) for r in self.rows] 

            title = self.column_titles.get(name) or name
            titles.append(titles)
            formatter = self.column_formatters.get(name)
            maxlen = len(title) # FIXME: Number of unicode characters as printed to console
            
            while True:
                strs = []
                redo = False
                for v in vals:  
                    if v is None:
                        s = None
                    if formatter:
                        s = formatter(v, maxlen=maxlen)
                        s =  f"{s}"
                        if len(s) > maxlen :
                            maxlen = len(s)
                            redo = True 
                    else:
                        s = f"{v}"
                        maxlen = max(maxlen, len(s))
                    strs.append(s)
                if not redo:
                    matrix.append(strs)
                    collen.append(maxlen)
                    break

        # Printing...
        for i,name in enumerate(self.columns):
            if i:
                print(" ")
            maxlen  = collen[i]
            print(f"{title:^{maxlen}}")
        for i,name in enumerate(self.columns):
            if i:
                print(" ")
            maxlen = collen[i]
            print("-" * maxlen)
        for j in range(nrows):
            for i,name in enumerate(self.columns):
                if i:
                    print(" ") 
                maxlen = collen[i]
                s = matrix[j][i]
                print(f"{s:^{maxlen}}")
            



class BenchVariants:
    def __init__(self, default_size, serial=None, cuda=None):
        None



class BenchResult:
    def __init__(self,name:str, wtime : datetime.timedelta, rtime : datetime.timedelta,acceltime: datetime.timedelta ,maxrss):
        self.name=name
        self.wtime=wtime
        self.rtime=rtime
        self.acceltime=acceltime
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
        acceltimesum = None
        count = len(benchmark)
        for it in benchmark:
            walltime = float(it.attrib['walltime'])
            wallsum += walltime
            usertime =float( it.attrib['usertime'])
            usersum += usertime
            kerneltime = float(it.attrib['kerneltime'])
            kernelsum += kerneltime
            if 'acceltime' in it.attrib:
                if acceltimesum is  None:
                    acceltimesum = 0
                acceltime  = float(it.attrib['acceltime'])
                acceltimesum += acceltime
        yield BenchResult(name=name,wtime=walltime/count,rtime=usersum/count,acceltime=None if acceltime is None else acceltimesum/count, maxrss=maxrss) 
        



def run_benchs(config:str=None,serial=[],cuda=[]):
    results = []
    for e in serial:
        results += list(run_gbench(exe=e))

    for e in cuda:
        results += list(run_gbench(exe=e))

    table = Table()
    def path_formatter(v:pathlib.Path, maxlen):
        return v.name
    def duration_formatter(v:pathlib.Path, maxlen):
        return v.name
    table.add_column('program', title="Benchmark", formatter=path_formatter)
    table.add_column('wtime', title="Wall time" formatter=duration_formatter)

    #print("Name: WallTime RealTime AccelTime MaxRSS")
    for r in results:
        table.add_row(program=r.name,wtime=r.wtime)
        #print(f"{r.name}: {r.wtime} {r.rtime} {r.acceltime} {r.maxrss}")

    table.print()



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
