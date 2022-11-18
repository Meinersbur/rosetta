#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from audioop import mul
from cmath import exp
from itertools import count
import sys
import argparse
import subprocess
import pathlib
import os
import datetime
import json
from typing import Iterable
import xml.etree.ElementTree as et
import colorama  
import math
import argparse
from collections import defaultdict
import io
import configparser
import typing 
import contextlib
import importlib
import importlib.util


# Not included batteries
import cwcwidth
#import tqdm # progress meter

# Rosetta-provided
import invoke
from support import *
from orderedset import OrderedSet
from cmdtool import *


# FIXME: Hack
colorama.Fore.BWHITE = colorama.ansi.code_to_chars(97)





class StrConcat: # Rename: Twine
    def __init__(self, args):
        self.args = list(args)

    def __add__(self, other):
        common = self.args + [other]
        return str_concat(*common)

    def printlength(self):
        return sum(printlength(a) for a in self.args)

    def normalize(self):
        # Optional: collapse nested StrConact
        # Optional: concat neighboring string

        # hoist StrAlign to topmost
        for i,a in enumerate(self.args):
            a = normalize(a)
            if isinstance(a,StrAlign):
                prefixlen = sum(printlength(a) for a in self.args[:i])
                return StrAlign(StrConcat(self.args[:i] + [a.s] + self.args[i+1:]), prefixlen+a.align)
        return self

    def consolestr(self):
        return ''.join(consolestr(a ) for a in self.args)



class StrColor:
    def __init__(self, s, style):
        self.s = s
        self.style = style

    def __add__(self, other):
        return str_concat(self,other)
    
    def printlength(self):
        return printlength(self.s)

    def normalize(self):
        a = normalize(self.s)
        if isinstance(a,StrAlign):
            return StrAlign(StrColor(a.s,self.style),a.align)
        if a is self.s:
            return self
        return StrColor(a, self.style)

    def consolestr(self):
        from colorama import Fore,Back,Style
        if self.style in {Fore.BLACK, Fore.RED, Fore.GREEN, Fore.YELLOW,Fore. BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE, Fore.BWHITE}:
            reset = Fore.RESET
        elif self.style in {Back.BLACK, Back.RED, Back.GREEN, Back.YELLOW,Back. BLUE, Back.MAGENTA, Back.CYAN, Back.WHITE}:
            reset = Back.RESET
        elif self.style in {Style.DIM, Style.NORMAL, Style.BRIGHT}:
            reset = Style.RESET_ALL
        else:
            reset = ''
        return self.style + consolestr(self.s) + reset



 


class StrAlign:
    LEFT= NamedSentinel('LEFT')
    CENTER= NamedSentinel('CENTER')
    RIGHT= NamedSentinel('RIGHT')

    # TODO: swap align/pos
    def __init__(self, s, align=None, pos=LEFT): 
        self.s = s
        self.align=align
        self.pos = pos

    def __add__(self, other):
        return str_concat(self,other)

    def normalize(self):
        # StrAlign cannot be nested
        return self

    def printlength(self):
        return printlength(self.s)




def str_concat(*args):
    return StrConcat(args=args)

def consolestr(s):
    if isinstance(s, str):
        return s
    return s.consolestr()

def printlength(s):
    if isinstance(s, str): 
        result = cwcwidth.wcswidth(s) 
        if result != len(s):
            pass
        return result
    return s.printlength()

def normalize(s):
    if isinstance(s, str):
        return s
    return s.normalize()




def default_formatter(v):
    return str(v)




# TODO: Support subcolumns
class Table:
    def __init__(self):
        self.columns = []
        self.supercolumns = dict()
        self.column_titles = dict()
        self.column_formatters = dict()
        self.rows = []

    def add_column(self, name, title=None, formatter=None):
        assert name not in self.columns,"Column names must be unique"
        self.columns.append(name)
        if title is not None:
            self.column_titles[name] = title
        if formatter is not None:
            self.column_formatters[name] = formatter

    def make_supercolumn(self, name, subcolumns):
        assert name in self.columns, "add column first before making it a supercolumn"
        assert name not in self.supercolumns, "already is a supercolumn"
        self.supercolumns[name] = subcolumns


    def add_row(self, **kwargs):
        self.rows .append(kwargs)

    def print(self):
        matrix = []
        nrows = len(self.rows)
        ncols = len(self.columns)

        colname_to_idx = dict()
        for i,name in enumerate(self.columns):
            colname_to_idx[name] = i

        name_to_leafidx = dict()
        leafidx_to_name=dict() #TODO: array
        for i,name in enumerate(name for name in self.columns if name not in self.supercolumns):
            name_to_leafidx[name] = i
            leafidx_to_name[i] = name

        # Determine columns and their max width
        collen = []
        colleft = []
        colright = []
        titles = []
        for i,name in enumerate(self.columns):
            vals = [r.get(name) for r in self.rows] 
            strs = []

            # TODO: Handle title just like another row
            title = self.column_titles.get(name) or name # TODO: Allow empty titles for supercolumns
            formatter = self.column_formatters.get(name) or default_formatter
            maxlen = printlength(title) 
            titles.append(title)
            maxleft  = 0
            maxright = 0
           
            for v in vals:  
                if v is None:
                    strs.append(None)
                else:
                    s = formatter(v) # TODO: support embedded newlines
                    s = normalize(s)
                    if isinstance(s,StrAlign):
                        l = printlength(s.s)
                        left = s.align
                        right = l - left
                        maxleft = max(maxleft,left)
                        maxright = max(maxright,right)
                    else:
                        l = printlength(s)
                        maxlen = max(maxlen,l)
                    strs.append(s)

            maxlen = max(maxlen, maxleft+maxright)
            collen.append(maxlen)
            colleft.append(maxleft)
            colright.append(maxright)
            matrix.append(strs)

        # Adapt for supercolumns
        # TODO: supercolumns might be hierarchical, so order is relevant TODO: determine the range of leaf columns in advance
        for supercol,subcols in self.supercolumns.items():
            subcollen = sum(collen[colname_to_idx.get(s)] for s in subcols)

            # print() inserts one space between items
            subcollen += len(subcols) -1
            supercollen = collen[colname_to_idx.get(supercol)]
            if subcollen < supercollen:
                # supercolumn is wider than subcolumns: divide additional space evenly between subcolumns
                overhang =  supercollen - subcollen
                for i,subcol in enumerate(subcols):
                    addlen = ((i+1)*overhang+len(subcols)//2)//len(subcols) - (i*overhang+len(subcols)//2)//len(subcols) 
                    collen[colname_to_idx.get(subcol)] += addlen 
            elif subcollen > supercollen:
                # subcolumns are wider than supercolumn: extend supercolumn
                collen[colname_to_idx.get(supercol)] = subcollen



        # Printing...
        def centering(s,collen):
            printlen = printlength(s)
            half = (collen - printlen)//2
            return ' '*half + consolestr(s) + ' ' *(collen - printlen - half)
        def raggedright(s,collen):
            printlen = printlength(s)
            return consolestr(s)  + ' ' * (collen - printlen)
        def aligned(s,maxlen,maxleft,alignpos):
            if alignpos is None:
                return raggedright(s,maxlen)
            else:
                printlen = printlength(s)
                indent = maxleft - alignpos
                cs = consolestr(s)
                return ' ' * indent + cs + ' '*(maxlen - printlen - indent)
        def linesep():
           print(*(colorama.Style.DIM + '-'*collen[colname_to_idx.get(colname)] + colorama.Style.RESET_ALL for i,colname in leafidx_to_name.items())) 

        def print_row(rowdata: dict):
            leafcolnum = len(name_to_leafidx)

            lines = [[' ' * collen[colname_to_idx.get( leafidx_to_name.get(j))]  for j in range(0,leafcolnum) ]]
            currow = [0] * leafcolnum

            def set_cells(celldata, supercol,cols):
                nonlocal lines,currow
                if not celldata:
                    return 
                indices = [name_to_leafidx.get(c) for c in cols]
                start = min(indices)
                stop = max(indices)
                for i in range(start,stop+1):
                    currow[i] += 1
                needlines = max(currow[cur] for cur in range(start,stop+1))
                while len(lines) < needlines:
                    lines.append([" " * collen[colname_to_idx.get( leafidx_to_name.get(j))]  for j in range(0,leafcolnum)])

                def colval(s,maxlen,maxleft,maxright):
                    #maxlen = collen[i]
                    #maxleft = colleft[i]

                    if isinstance(s,StrAlign):
                        if s.pos == StrAlign.LEFT:
                            return aligned(s.s,maxlen,maxleft,s.align)
                        elif s.pos == StrAlign.CENTER:
                            if s.align is None:
                                return centering(s.s, maxlen) 
                            # TODO: check correctness
                            printlen = printlength(s)
                            rightindent = maxright - printlen - s.align
                            return centering( consolestr(s) + ' '*rightindent)
                        elif s.pos == StrAlign.RIGHT:
                            if s.align is None:
                                return raggedright(s, maxlen) 
                            # TODO: check correctness
                            printlen = printlength(s)
                            rightindent = maxright - printlen - s.align
                            leftindent = maxlen - rightindent - printlen
                            return ' '*leftindent + consolestr(s) + ' ' *rightindent


                    # Left align by default
                    return raggedright(s,maxlen)

                totallen = sum( collen[colname_to_idx.get(leafidx_to_name.get( j )  ) ] for j in range(start,stop+1)) + stop - start
                totalleft = colleft[colname_to_idx.get(supercol)]
                totalright = colright[colname_to_idx.get(supercol)]
                lines[needlines-1][start] = colval(celldata,totallen,  totalleft,totalright)
                for i in range(start+1,stop+1):
                    lines[needlines-1][i] = None

            for supercol,subcols in self.supercolumns.items():
                superdata = rowdata.get(supercol)
                set_cells(superdata, supercol, subcols)

            for i,colname in leafidx_to_name.items():
                celldata = rowdata.get(colname)
                set_cells( celldata, colname, [colname ])

            for line in lines :
                print(*(l for l in line if l is not None))



        print()
        print_row(self.column_titles)
        linesep()

        for j in range(nrows):
            print_row({ colname: matrix[i][j] for colname,i in colname_to_idx.items() } )

        return 
        print()
        print(*(centering(titles[i],collen[i]) for i in range(ncols)))
        linesep()

        for j in range(nrows):
            def colval(i,name):
                maxlen = collen[i]
                maxleft = colleft[i]
                s = matrix[i][j]
               
                if s is None:
                    return ' ' * maxlen
                if isinstance(s,StrAlign):
                    return aligned(s.s,maxlen,maxleft,s.align)

                # Left align by default
                return raggedright(s,maxlen)

            print(*(colval(i,name) for i,name in enumerate(self.columns)))
        #linesep()





# Summary statistics
# Don't trust yet, have to check correctness
# TODO: Use numpy
class Statistic:
    # TODO: Consider using SciPy's scipy.distributions.t.cdf if available
    # Or just directly use scipy.distributions.ttest_ind
    # https://pythonguides.com/scipy-confidence-interval/
    studentt_density_95 = list( {
1: 12.706, 
2: 4.303, 
3: 3.182, 
4: 2.776, 
5: 2.571, 
6: 2.447, 
7: 2.365, 
8: 2.306, 
9: 2.262, 
10: 2.228, 
11: 2.201, 
12: 2.179, 
13: 2.160, 
14: 2.145, 
15: 2.131, 
16: 2.120, 
17: 2.110, 
18: 2.101, 
19: 2.093, 
20: 2.086, 
21: 2.080, 
22: 2.074, 
23: 2.069,
24: 2.064, 
25: 2.060, 
26: 2.056, 
27: 2.052, 
28: 2.048, 
29: 2.045, 
30: 2.042, 
35: 2.030, 
40: 2.021,
45: 2.014, 
50: 2.009, 
60: 2.000, 
70: 1.994, 
80: 1.990, 
90: 1.987, 
100: 1.984, 
150: 1.976, 
200: 1.972, 
250: 1.969, 
300: 1.968, 
400: 1.966, 
500: 1.965, 
600: 1.964, 
800: 1.963, 
1000: 1.962, 
100000: 1.960, 
    }.items())

    def __init__(self,samples,sum,sumabs,sumsqr,sumreciproc,geomean):
        self._samples = list(samples) # Assumed sorted
        self._sum = sum
        self._sumabs = sumabs
        self._sumsqr = sumsqr
        self._sumreciproc = sumreciproc
        self._geomean = geomean
 
    @property
    def samples (self):
        return self._samples

    @property
    def is_empty(self):
        return not not self._samples

    @property
    def count(self):
        return len(self._samples)

    @property
    def minimum(self):
        return self._samples[0]

    @property
    def maximum(self):
        return self._samples[-1]


    # Location

    @property
    def mean(self):
        return self._sum/self.count

    @property
    def geomean(self):
        return self._geomean

    @property
    def harmonicmean(self):
        return self.count  / self._sumreciproc  

    @property
    def median(self):
        return self.quantile(1,2)

    def quantile(self,k,d):
        assert d >=1
        assert k >= 0 <= d
  
        if not self._vals:
            return None

        if k == 0:
            return self._vals[0]
        if k == n:
            return self._vals[-1]

        n = self.count
        if (k*n-1) % d == 0:
            return self._vals[(k*n-1)//d]
    

    def quartile(self,k:int):
        return self.quantile(k,4)

    def decile(self,k:int):
        return self.quantile(k,10)

    def percentile(self,k:int):
        return self.quantile(k,100)

    @property 
    def midrange(self):
        return (self.minimum + self.maximum)/2

    def mode(self, boxsize):
        boxes = defaultdict(0)
        for v in self._samples:
            boxidx = round(v / boxsize)
            boxes[boxidx] += 1

        maxcount = 0
        boxids = []
        for boxidx,count in boxes.items():
            if count < maxcount:
                continue
            if count > maxcount:
                maxcount = count
                boxids.clear()
                continue
            boxids.append(boxids)
            
        if len(boxids) % 2 == 0:
            midboxid = (boxids[len(boxids)//2] + boxids[len(boxids)//2+1])/2
        else:
            midboxid = boxids[len(boxids)//2] 
        return midboxid * boxsize


    # Spread

    @property
    def variance(self):
        def sqr(x):
            return x * x
        n = self.count
        return self._sumsqr / n - sqr(self.mean)

    @property
    def corrected_variance(self):
        n = self.count
        return self.variance * n / (n-1)


    @property
    def stddev(self):
        return math.sqrt(self.variance)

    @property
    def corrected_stddev(self):
        return math.sqrt(self.corrected_variance)

    @property
    def relative_stddev(self):
        return self.stddev / abs(self.mean)

    @property 
    def range(self):        
        return self.maximum - self.minimum

    # Mean squared error/deviation
    @property 
    def mse(self):
        def sqr(x):
            return x * x
        e = self.median # Using median as estimator
        return  sum(sqr(x - e) for x in self._vals) / self.count

    # Root mean squared error/deviation
    @property 
    def rmse(self):
        return math.sqrt(self.mse)

    @property 
    def relative_rmse(self):
        return self.rmse / abs(self.median)

    # Mean absolute error/deviation
    @property 
    def mad(self):
        median = self.median
        return sum(abs(x - median) for x in self._vals) / self.count

    # Symmetric confidence interval around mean, assuming normal distributed samples
    # TODO: Asymetric confidence interval
    def abserr(self,ratio=0.95):
        assert ratio == 0.95, r"Only supporting two-sided 95% confidence interval"
        n = self.count
        if n < 2:
            return None

        # Table lookup
        # TODO: bisect
        c = None
        for (n1,p1),(n2,p2) in zip(Statistic.studentt_density_95, Statistic.studentt_density_95[1:]):
            if n1 <= n <= n2:
                # Linear interpolation
                r = (n - n1) / (n2 - n1)
                c = r * p2 + (1 - r) * p1
                break
        if c is None:
            c = Statistic.studentt_density_95[-1][1]

        return c * self.corrected_variance / math.sqrt(n)

    def relerr(self,ratio=0.95):
        mean = self.mean
        if not mean:
            return None
        return self.abserr(ratio=ratio) / self.mean

    # TODO: signal/noise ratio (relative_rmse?)





def statistic(data):
    vals = sorted(d for d in data if d is not None)
    n = 0
    hasnonpos = False
    sum = 0
    sumabs = 0
    sumsqr = 0
    sumreciproc = 0
    prod = 1
    for v in vals:
        if v <= 0:
            hasnonpos = True
        sum += v
        sumabs = abs(v)
        sumsqr += v*v
        if not hasnonpos:
            sumreciproc += 1//v
            prod *= v
        n += 1
    
    if hasnonpos:
        geomean = None
        sumreciproc = None
    else:
        geomean = prod**(1/n)
    
    
    return Statistic(samples=vals,sum=sum,sumabs=sumabs,sumsqr=sumsqr,geomean=geomean,sumreciproc=sumreciproc)



class BenchVariants:
    def __init__(self, default_size, serial=None, cuda=None):
        None



#TODO: dataclass
class BenchResult:
    def __init__(self, name:str, ppm:str,buildtype:str,configname:str, count:int, durations, maxrss=None, cold_count=None, peak_alloc=None):
        #self.bench=bench
        self.name=name
        self.ppm = ppm
        self.buildtype = buildtype
        self.configname = configname
        self.count=count
        #self.wtime=wtime
        #self.utime=utime
        #self.ktime=ktime
        #self.acceltime=acceltime
        self.durations=durations
        self.maxrss=maxrss
        self.cold_count = cold_count
        self.peak_alloc = peak_alloc



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


def name_or_list(data):
    if not data :
        return None
    if isinstance(data , str):
        return data
    if not isinstance(data , Iterable):
        return data
    if len(data)==1 :
        return data[0]
    return data



# TODO: dataclass?
class BenchResultGroup:
    def __init__(self,results):
        self.name = name_or_list(unique(r.name for r in results))
        self.ppm = name_or_list(unique(r.ppm for r in results))
        self.buildtype =name_or_list( unique(r.buildtype for r in results))
        self.configname = name_or_list(unique(r.configname for r in results))

        # Combine all durations to a single statistic; TODO: Should we do something like mean-of-means?
        measures = unique(k for r in results for k in r.durations.keys())
        self.durations = { m : statistic( v for r in results for v in r.durations[m]._samples )    for m in measures  }



# TODO: enough precision for nanoseconds?
# TODO: Use alternative duration class
def parse_time(s:str):
    if s.endswith("ns"):
        return float(s[:-2]) / 1000000000
    if s.endswith("us") or s.endswith("µs") :
        return float(s[:-2]) / 1000000
    if s.endswith("ms")  :
        return float(s[:-2]) / 1000
    if s.endswith("s")  :
        return float(s[:-1]) 
    if s.endswith("m")  :
        return float(s[:-1])   * 60  
    if s.endswith("h")  :
        return float(s[:-1])   * 60 * 60 
    raise Exception("Don't know the duration unit")


# TODO: Recognize Kibibytes
def parse_memsize(s:str):
    if s.endswith("K") :
            return  math.ceil( float(s[:-1]) * 1024)
    if s.endswith("M") :
            return  math.ceil(float(s[:-1]) * 1024* 1024)
    if s.endswith("G") :
            return  math.ceil( float(s[:-1]) * 1024* 1024* 1024)
    return int(s)


def do_run(bench,args,resultfile) :
    exe = bench.exepath 

    start = datetime.datetime.now()
    args.append(f'--xmlout={resultfile}')
    print("Executing", shjoin([exe] + args))
    #p = subprocess.Popen([exe] + args ,stdout=subprocess.PIPE,universal_newlines=True)
    p = subprocess.Popen([exe] + args)
    #stdout = p.stdout.read()
    unused_pid, exitcode, ru = os.wait4(p.pid, 0) # TODO: Integrate into invoke TODO: Fallback on windows TODO: should measure this in-process
    

    stop = datetime.datetime.now()
    p.wait() # To let python now as well that it has finished

    assert resultfile.is_file(), "Expecting result file to be written by benchmark"

    wtime = max(stop - start,datetime.timedelta(0))
    utime = ru.ru_utime
    stime = ru.ru_stime
    maxrss = ru.ru_maxrss * 1024
    return resultfile


def path_formatter(v:pathlib.Path):
    if v is None:
        return None
    return StrColor(pathlib.Path(v).name,colorama.Fore.GREEN)

def duration_formatter(best=None,worst=None):
        def formatter(s: Statistic):
            if s is None:
                return None
            v = s.mean
            d = s.relerr()
            def highlight_extremes(s):
                if best is not None and worst is not None and best < worst:
                    if v <= best:
                        return StrColor(s, colorama.Fore.GREEN)
                    if v >= worst:
                        return StrColor(s, colorama.Fore.RED)
                return s
        
            if d and d >= 0.0001:
                errstr = f"(±{d:.1%})"
                if d >= 0.02:
                    errstr = StrColor(errstr, colorama.Fore.RED)
                errstr = str_concat(' ',errstr)
            else:
                errstr = ''

            if v >= 1:
                return highlight_extremes(align_decimal(f"{v:.2}")) +StrColor("s", colorama.Style.DIM) + (str_concat(' ', errstr) if errstr else '')
            if v*1000 >= 1:
                return highlight_extremes(align_decimal(f"{v*1000:.2f}") )+ StrColor("ms", colorama.Style.DIM) + errstr
            if v*1000*1000 >= 1:
                return highlight_extremes(align_decimal(f"{v*1000*1000:.2f}")) + StrColor("µs", colorama.Style.DIM) + errstr
            return highlight_extremes(align_decimal(f"{v*1000*1000*1000:.2f}")) + StrColor( "ns", colorama.Style.DIM) + errstr
        return formatter


def load_resultfiles(resultfiles,filterfunc=None):
    results = []
    for resultfile in resultfiles:
        benchmarks = et.parse(resultfile).getroot()

        for benchmark in benchmarks:
            name = benchmark.attrib['name']
            n = benchmark.attrib['n']
            cold_count = benchmark.attrib.get('cold_iterations') 
            peak_alloc = int( benchmark.attrib.get('peak_alloc'))
            maxrss =int( benchmark.attrib.get('maxrss'))
            ppm  = benchmark.attrib.get('ppm')
            buildtype  = benchmark.attrib.get('buildtype')
            configname = benchmark.attrib.get('configname')
            count = len(benchmark)

            time_per_key = defaultdict(lambda :  [])
            for b in benchmark :
                for k, v in b.attrib.items():
                    time_per_key[k] .append(parse_time(v))

            stat_per_key = {}
            for k,data in time_per_key.items():
                stat_per_key[k] = statistic(data)

            item = BenchResult( name=name, ppm=ppm, buildtype=buildtype, count=count,durations=stat_per_key, cold_count=cold_count,peak_alloc=peak_alloc,configname=configname,maxrss=maxrss) 
            if filterfunc and not filterfunc(item):
                    continue
            results.append( item)
    return results


def evaluate(resultfiles):
    results = load_resultfiles(resultfiles)

    stats_per_key = defaultdict(lambda :  [])
    for r in results:
        for k,stat in r.durations.items():
            stats_per_key[k] .append(stat)

    summary_per_key = {} # mean of means
    for k,data in stats_per_key.items():
        summary_per_key[k] = statistic(d.mean for d in data)


    table = Table()
    def count_formatter(v:int):
        s = str(v)
        return StrAlign( StrColor(str(v),colorama.Fore.BLUE), printlength(s))
    def ppm_formatter(s:str):
        return getPPMDisplayStr(s)


    table.add_column('program', title=StrColor("Benchmark", colorama.Fore.BWHITE), formatter=path_formatter)
    table.add_column('ppm', title="PPM",formatter=ppm_formatter)
    table.add_column('buildtype', title="Buildtype")
    table.add_column('n', title=StrColor("Repeats", colorama.Style.BRIGHT),formatter=count_formatter)
    for k,summary in summary_per_key.items():
        table.add_column(k, title=StrColor(getMeasureDisplayStr(k), colorama.Style.BRIGHT),formatter=duration_formatter(summary.minimum,summary.maximum))


    for r in results:
        # TODO: acceltime doesn't always apply
        table.add_row(program=r.name , ppm=r.ppm, buildtype=r.buildtype, n=r.count,**r.durations)

    table.print()



def get_column_data(result: BenchResult, colname:str):
    if result is None:
        return None
    if colname == "program":
        return result.name
    if colname == "ppm":
        return result.ppm
    if colname == "buildtype":
        return result.buildtype
    if colname == "configname":
        return result.configname
    if colname == "walltime":
        return result.durations.get("walltime")
    assert False, "TODO: Add to switch of use getattr"



def print_comparison(groups_of_results, list_of_resultnames, common_columns=["program"], compare_columns=[]):
    table = Table()

    # Transpose list of lists
    #results_of_groups = [] 
    #for group in groups_of_results:
    #    if len(group) > len(results_of_groups):
    #        results_of_groups.extend([[]] * (len(group)  - len(results_of_groups)) )
    #    for i,result in enumerate(group):
    #        results_of_groups[i].append(result)
        

    for col in common_columns:
        if col == "program": 
            table.add_column(col, title= StrAlign( StrColor("Benchmark", colorama.Fore.BWHITE),pos=StrAlign.CENTER), formatter=path_formatter)
        else: # TODO: proper column name
            table.add_column(col, title= StrAlign( StrColor(col, colorama.Fore.BWHITE),pos=StrAlign.CENTER))

    for j,col in enumerate(compare_columns):
        supercolumns = []
        table.add_column(col, StrAlign( StrColor(getMeasureDisplayStr(col), colorama.Style.BRIGHT)  ,  pos=StrAlign.CENTER))
        for i, resultname in enumerate(list_of_resultnames): # Common title
            sol = f"{col}_{i}"
            supercolumns.append(sol)
            table.add_column(sol, title=StrAlign( StrColor(resultname, colorama.Style.BRIGHT),pos=  StrAlign.CENTER) ,formatter=duration_formatter())
        table.make_supercolumn(f"{col}", supercolumns)


    for result in groups_of_results:
        representative = result[0] # TODO: collect all occuring group values
        data = dict()
        for col in common_columns:
            data[col] = get_column_data(representative, col)
        for col in compare_columns:
            for i, resultname in enumerate(list_of_resultnames):
                data[f"{col}_{i}"] = get_column_data(result[i], col)
        table.add_row(**data)

    table.print()



def compareby(results : Iterable[BenchResult], compare_by: str):
    results_by_group =  defaultdict(lambda :  [])
    for result in results:
        cmpval = get_column_data(result, compare_by)
        results_by_group[cmpval].append(result)
    return results_by_group



def grouping(results : Iterable[BenchResult], compare_by: str,  group_by=None): 
    # TODO: allow compare_by multiple columns
    # TODO: allow each benchmark to be its own group; find description for each such "group"
    results_by_group = defaultdict(lambda :  defaultdict(lambda :  []))
    all_cmpvals = OrderedSet()
    for result in results:
        group = tuple(get_column_data(result, col) for col in group_by)
        cmpval = get_column_data(result, compare_by)
        all_cmpvals.add(cmpval)
        results_by_group[group][cmpval].append(result)

    grouped_results = []
    all_groups = []
    for group,group_results in results_by_group.items():
        group_cmp_results = []
        for cmpval in all_cmpvals:
            myresults = group_results.get(cmpval)
            if myresults:
                group_cmp_results.append(BenchResultGroup(myresults))
            else:
                # No values
                group_cmp_results.append(None)
            #is_unique_groups = tuple(same_or_none( g[i] for g in groups) is not None for  i in range(len(group_by)))
        grouped_results.append(group_cmp_results)
        all_groups.append(group)

    # Find all fields that could be grouped by and have different values
    show_groups = divergent_fields(group_by,results)

    return grouped_results,list(all_cmpvals),show_groups
    

def divergent_fields(group_by,results):
    show_groups = []
    for col in group_by:
        common_value=None
        has_different_values= False
        for result in results:
            val = get_column_data(result, col)
            if common_value is None:
                common_value = val
            elif common_value == val:
                continue
            else:
                has_different_values = True
                break
        if has_different_values:
            show_groups.append(col)
    return show_groups



def results_compare(resultfiles: list,compare_by,group_by=None,compare_val=None,show_groups=None):
    results = load_resultfiles(resultfiles)

    # Categorical groupings
    if group_by is None:
        group_by = ["program", "ppm", "buildtype", "configname"]
        group_by.remove(compare_by)

    grouped_results, all_cmpvals, div_groups = grouping(results,compare_by=compare_by,group_by=group_by)

    print_comparison(groups_of_results=grouped_results, list_of_resultnames=all_cmpvals, common_columns=show_groups or div_groups, compare_columns=compare_val)





def results_boxplot(resultfiles: list,group_by=None,compare_by=None,filterfunc=None):
    r"""Produce a boxplot for benchmark results

    :param group_by:   Summerize all results that have the same value for these properties. No summerization if None.
    :param compare_by: Which property to compare side-by-side in a group of plots. Implicitly enables grouping.
    """
    results = load_resultfiles(resultfiles,filterfunc=filterfunc)

    if group_by or compare_by:
        if group_by is None:
            group_by = ["program", "ppm", "buildtype", "configname"]
        if compare_by:
            group_by.remove(compare_by)

        grouped_results,all_cmpvals,div_groups = grouping(results, compare_by='configname',group_by=group_by) 
        groupdata  = [[b.durations['walltime'].samples for b in group] for group in grouped_results] 
    else:
        # Each result in its own group
        grouped_results = [[r] for r in results]
        div_groups = divergent_fields(["program", "ppm", "buildtype", "configname"],results)
        all_cmpvals = [""]

    def make_label(g: tuple):
        first = g[0] 
        return ', '.join( get_column_data(first, k ) for k in div_groups )
    labels = [make_label(g) for g in grouped_results]

    import matplotlib.colors as mcolors
    from cycler import cycler
    import matplotlib.pyplot as plt

    left=1
    right=0.5
    numgroups  = len(grouped_results)
    benchs_per_group = len(all_cmpvals)
    barwidth = 0.3
    groupwidth = 0.2 + benchs_per_group * barwidth
    width = left + right + groupwidth*numgroups
    fig, ax = plt.subplots(figsize=(width, 10))
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [c['color'] for j,c in zip( range(benchs_per_group),prop_cycle)] # TODO: Consider seaborn palettes

    fig.subplots_adjust(left=left/width, right=1-right/width, top=0.95, bottom=0.25)


    for i, group in enumerate(grouped_results):
        benchs_this_group = len(group)           
        for j, benchstat in enumerate(group): # TODO: ensure grouped_results non-jagged so colors match
            data = benchstat.durations['walltime'].samples  # TODO: Allow other datum that walltime
            rel = (j-benchs_this_group/2.0+0.5)*barwidth
            box = ax.boxplot(data, positions=[i*groupwidth + rel], 
                            notch=True,showmeans=False, showfliers=True,sym='+',
                            widths=barwidth,
                                patch_artist=True,  # fill with color
                            )
            for b in box['boxes']:
                    b.set_facecolor(colors[j])
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',  alpha=0.5)

    for j,(c,label) in enumerate(zip(colors,all_cmpvals)):
        # Dummy item to add a legend handle; like seaborn does
        rect = plt.Rectangle([0, 0], 0, 0,
                            # linewidth=self.linewidth / 2,
                            # edgecolor=self.gray,
                            facecolor=c,
                            label=label)
        ax.add_patch(rect)


    # TODO: Compute conf_intervals consistently like the table, preferable using the student-t test.
    #x.grid(linestyle='--',axis='y')
    ax.set(
        axisbelow=True,  # Hide the grid behind plot objects
        xlabel='Benchmark',
        ylabel='Walltime [s]',
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xticks([groupwidth*i for i in range(len(labels))])
    ax.set_xticklabels(labels,rotation=20,ha="right",rotation_mode="anchor")
    
    plt.legend()

    #for label in ax.get_xticklabels(): # https://stackoverflow.com/a/43153984
    #    label.set_ha("right")
    #    label.set_rotation(45)
    return plt.gcf()




def run_gbench(bench,problemsizefile,resultfile):
    args = []
    if problemsizefile:
        args.append(f'--problemsizefile={problemsizefile}')
    return do_run(bench=bench, args=args,resultfile=resultfile)
        


def align_decimal(s):
    # FIXME: Don't align in scientific notation?
    pos = s.find('.')
    if pos >= 0:
        return StrAlign(s, pos)
    return StrAlign(s, printlength(s))



def getMeasureDisplayStr(s:str):
   return  {'walltime': "Wall", 'usertime': "User", 'kerneltime': "Kernel", 
   'acceltime': "CUDA Event", 
   'cupti': "nvprof", 'cupti_compute': "nvprof Kernel", 'cupti_todev': "nvprof H->D", 'cupti_fromdev': "nvprof D->H"}.get(s, s)


def getPPMDisplayStr(s:str):
    return {'serial': "Serial", 'cuda': "CUDA", 'omp_parallel': "OpenMP parallel", 'omp_task' : "OpenMP task", 'omp_target': "OpenMP Target Offloading"}.get(s,s)




class Benchmark:
    def __init__(self,basename,target,exepath,buildtype,ppm,configname,sources=None,benchpropfile=None,compiler=None,compilerflags=None,pbsize=None,benchlistfile=None,is_ref=None):
        self.basename = basename
        self.target=target
        self.exepath =exepath 
        self.buildtype=buildtype
        self.ppm = ppm
        self.configname = configname
        self.sources= [mkpath(s) for s in sources] if sources else None
        self.benchpropfile=benchpropfile
        self.compiler = mkpath(compiler)
        self.compilerflags=compilerflags
        self.pbsize = pbsize # default problemsize
        self.benchlistfile = benchlistfile 
        self.is_ref = is_ref 

    @property 
    def name(self):
        return self.basename


def get_problemsizefile(srcdir=None, problemsizefile=None):
    if problemsizefile:
        if not problemsizefile.is_file():
            # TODO: Embed default sizes
            die(f"Problemsize file {problemsizefile} does not exist.",file=sys.stderr)
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



def get_refpath(bench,refdir,problemsizefile):
    pbsize = get_problemsize(bench,problemsizefile=problemsizefile)
    reffilename = f"{bench.name}.{pbsize}.reference_output"
    refpath = refdir/reffilename
    return refpath




def ensure_reffile(bench: Benchmark,refdir,problemsizefile):
    refpath = get_refpath(bench,refdir=refdir,problemsizefile=problemsizefile)

    if refpath.exists():
        # Reference output already exists; check that it is the latest
        benchstat = bench.exepath.stat()
        refstat  = refpath.stat()
        if benchstat.st_mtime < refstat.st_mtime:
            print(f"Reference output of {bench.name} already exists at {refpath} an is up-to-date")
            return 
        print(f"Reference output {refpath} an is out-of-date")
        refpath.unlink()

    # Invoke reference executable and write to file
    args = [bench.exepath, f'--verify',  f'--verifyfile={refpath}']
    if problemsizefile:
        args.append(f'--problemsizefile={problemsizefile}')
    invoke.call(*args, print_command=True)    
    if not refpath.is_file():
        print(f"{refpath} not been written?")
        assert refpath.is_file()

    print(f"Reference output of {bench.name} written to {refpath}")
 


def ensure_reffiles(refdir,problemsizefile,filterfunc=None,srcdir=None):
    problemsizefile = get_problemsizefile(srcdir=srcdir,problemsizefile=problemsizefile)
    for bench in benchmarks:
        if filterfunc and not filterfunc(bench):
            continue
        ensure_reffile(bench,refdir=refdir,problemsizefile=problemsizefile)


# math.prod only available in Python 3.8
def prod(iter):
    result = 1
    for v in iter:
        result *=v
    return result


def run_verify(problemsizefile,filterfunc=None,srcdir=None,refdir=None):
    problemsizefile = get_problemsizefile(srcdir=srcdir,problemsizefile=problemsizefile)

    #x = request_tempdir(prefix=f'verify') 
    #tmpdir = mkpath(x.name)
    refdir.mkdir(exist_ok=True,parents=True)
        
    for e  in benchmarks:
        if filterfunc and not filterfunc(e):
            continue

        ensure_reffile(e, refdir=refdir, problemsizefile=problemsizefile)

        exepath = e.exepath
        refpath = get_refpath(e,refdir=refdir,problemsizefile=problemsizefile)
        pbsize = get_problemsize(e,problemsizefile=problemsizefile)

    
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

               refspec,refdata = refline.split(':',maxsplit = 1)     
               refspec = refspec.split()
               refkind = refspec[0]
               refformat = refspec[1]
               refdim = int(refspec[2])
               refshape =  [int(i) for i in refspec[3:3+refdim]]
               refname = refspec[3+refdim] if len(refspec ) > 3+refdim else None
               refcount = prod(refshape)

               refdata = [float(v) for v in refdata.split()]
               if refcount != len(refdata):
                die(f"Unexpected array items in {refname}: {refcount} vs {len(refdata)}")
               
            
               testspec,testdata = testline.split(':',maxsplit = 1)  
               testspec = testspec.split()
               testkind = testspec[0]
               testformat = testspec[1]
               testdim = int(testspec[2])
               testshape =  [int (i) for i in testspec[3:3+testdim]]
               testname = testspec[3+testdim] if len(testspec ) > 3+testdim  else None
               testcount = prod(testshape) 
               
               testdata = [float(v) for v in testdata.split()]
               if testcount != len(testdata):
                die(f"Unexpected array items in {testname}: {testcount} vs {len(testdata)}")
               
               if refname is not None and testname is not None and refname != testname:
                  die(f"Array names {refname} and {testname} disagree")

               for i,(refv,testv) in enumerate(zip(refdata,testdata)):
                  coord = [str((i // prod(refshape[0:j])) % refshape[j]) for j in range(0,refdim)]
                  coord = '[' + ']['.join(coord) + ']'

                  if math.isnan(refv) and  math.isnan(testv):
                        print(f"WARNING: NaN in both outputs at {refname}{coord}")
                        continue
                  if math.isnan(refv): 
                    die(f"Array data mismatch: Ref contains NaN at {refname}{coord}")
                  if math.isnan(testv): 
                    die(f"Array data mismatch: Output contains NaN at {testname}{coord}")
                    

                  mid =  (abs(refv)+abs(testv))/2
                  absd = abs(refv-testv)
                  if mid == 0:
                    reld = 0 if absd==0 else math.inf 
                  else:
                    reld = absd/mid
                  if reld > 1e-4: # TODO: Don't hardcode difference
                    print(f"While comparing {refpath} and {testoutpath}:")
                    die(f"Array data mismatch: {refname}{coord} = {refv} != {testv} = {testname}{coord} (Delta: {absd}  Relative: {reld})")

        print(f"Output of {e.exepath} considered correct")       

               





def make_resultssubdir(within=None):
    global resultsdir
    within = within or resultsdir
    assert within
    now = datetime.datetime.now()
    i = 0
    suffix=''
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
        results .append(run_gbench(e,problemsizefile=problemsizefile,resultfile=thisresultdir))
    return results








def custom_bisect_left(lb, ub, func):
    assert ub >= lb
    while True:
        if lb == ub:
            return lb
        mid = (lb + ub + 1 ) // 2
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


mytempdir =None
globalctxmgr  = contextlib. ExitStack()
def request_tempdir(subdir=None): 
    global mytempdir
    if mytempdir :
        return mytempdir
    x = tempfile.TemporaryDirectory(prefix=f'rosetta-') # TODO: Option to not delete / keep in current directory
    mytempdir = mkpath(globalctxmgr.enter_context(x))
    return  mytempdir

def request_tempfilename(prefix=None,suffix=None,subdir=None): 
    tmpdir = request_tempdir(subdir=subdir)
    candidate =  tmpdir/  f'{prefix}{suffix}'
    i = 0
    while candidate.exists():
        candidate = tmpdir /  f'{prefix}-{i}{suffix}'
        i+=1

    return candidate

# TODO: merge with run_gbench
# TODO: repeats for stability
def probe_bench(bench:Benchmark, limit_walltime, limit_rss, limit_alloc):
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
        resultfile = request_tempfilename(subdir='probe', prefix=f'{bench.target}-pbsize{n}',suffix='.xml')
        do_run(bench,args=[f'--pbsize={n}', '--repeats=1'], resultfile=resultfile )
        [result ]= load_resultfiles([resultfile])
        if is_too_large(result):
            return -1
        return 1


    while func(n) != -1:
        lower_n = n
        n *= 2

    return custom_bisect_left(lower_n, n-1, func)
        
  


def run_probe(problemsizefile, limit_walltime, limit_rss, limit_alloc):
    if not problemsizefile:
        die("Problemsizes required")  

    problemsizecontent = []
    for bench in benchmarks:
        n = probe_bench(bench=bench, limit_walltime=limit_walltime,limit_rss=limit_rss,limit_alloc=limit_alloc)

        problemsizecontent.extend(
                [f"[{bench.name}]",
                f"n={n}",
                ""
                ]
        )
    with problemsizefile.open(mode='w+') as f:
        for line in problemsizecontent:
            print(line,file=f)



def runner_main(builddir):
    runner_main_run()

def runner_main_run(srcdir,builddir):
    with  globalctxmgr :
        parser = argparse.ArgumentParser(description="Benchmark runner", allow_abbrev=False)
        add_boolean_argument(parser, 'buildondemand', default=True, help="build to ensure executables are up-to-data")
        resultdir= builddir/ 'results'
        subcommand_run(parser,None,srcdir,builddirs=[builddir],refbuilddir=builddir,resultdir=resultdir)
        args = parser.parse_args(sys.argv[1:])

        subcommand_run(None,args,srcdir,builddirs=[builddir],buildondemand=args.buildondemand,refbuilddir=builddir,resultdir=resultdir)




def subcommand_run(parser,args,srcdir,buildondemand=False,builddirs=None,refbuilddir=None,filterfunc=None,resultdir=None):
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

        add_boolean_argument(parser, 'evaluate', default=None, help="Evaluate result")

        parser.add_argument('--boxplot', type=pathlib.Path, help="Save as boxplot to FILENAME")


    if args:
        # If neither no action is specified, enable --bench implicitly unless --no-bench
        probe = args.probe
        verify = args.verify
        bench =  args.bench
        evaluate = args.evaluate
        if bench is None and not verify and not probe:
            bench = True


        if probe:
            assert args.problemsizefile , "Requires to set a problemsizefile to set"
            run_probe(problemsizefile=args.problemsizefile, limit_walltime=args.limit_walltime, limit_rss=args.limit_rss, limit_alloc=args.limit_alloc)


        if verify:
            refdir = refbuilddir / 'refout'
            run_verify(problemsizefile=args.problemsizefile,refdir=refdir)

        if bench:
            resultfiles = run_bench(srcdir=srcdir,problemsizefile=args.problemsizefile,resultdir=resultdir)

            if args.boxplot:
                fig = results_boxplot(resultfiles)
                fig.savefig(fname=args.boxplot)
                fig.canvas.draw_idle() 

            if len(builddirs)> 1:
                results_compare(resultfiles, compare_by="configname", compare_val=["walltime"])
            else:
                evaluate(resultfiles)


        if evaluate:
           if not resultfiles:
                assert False, "TODO: Lookup last (successful) results dir"
            if len(configs) == 1:
                runner.evaluate(resultfiles)
            else:
                runner.results_compare(resultfiles, compare_by="configname", compare_val=["walltime"])




# TODO: Integrate into subcommand_run
def runner_main_verify(builddir,srcdir):
    parser = argparse.ArgumentParser(description="Benchmark verification", allow_abbrev=False)
    parser.add_argument('--problemsizefile', type=pathlib.Path, help="Problem sizes to use (.ini file)")
    add_boolean_argument(parser, 'buildondemand', default=True) # TODO: implement

    args = parser.parse_args()

    refdir = builddir / 'refout'
    return run_verify(problemsizefile=args.problemsizefile,refdir=refdir)





def  runner_main_probe(builddir):
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
benchmarks : typing .List[Benchmark]  =[]
def register_benchmark(basename,target,exepath,buildtype,ppm,configname,benchpropfile=None,compiler=None,compilerflags=None,pbsize=None):
    bench = Benchmark(basename=basename,target=target,exepath=mkpath(exepath), buildtype=buildtype,ppm=ppm,configname=configname,benchpropfile=benchpropfile,compiler=compiler,compilerflags=compilerflags,pbsize=pbsize,benchlistfile=benchlistfile,is_ref=import_is_ref )
    benchmarks.append(bench)



def load_register_file(filename,is_ref=False):
    global benchlistfile ,import_is_ref
    import importlib

    filename = mkpath(filename)
    benchlistfile = filename
    import_is_ref = is_ref 
    try:
        spec = importlib.util.spec_from_file_location(filename.stem, str(filename))
        module =  importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        benchlistfile = None
        import_is_ref = None



def gen_reference(exepath,refpath,problemsizefile):
    #print(f"{exepath=} {refpath=}")
    args = [exepath, f'--verify', f'--problemsizefile={problemsizefile}']
    invoke.call(*args, stdout=refpath,print_stderr=True,print_command=True)



def main(argv):
    colorama.init()
    parser = argparse.ArgumentParser(description="Benchmark runner", allow_abbrev=False)
    parser.add_argument('--gen-reference', nargs=2, type=pathlib.Path,  help="Write reference output file")
    parser.add_argument('--problemsizefile',  type=pathlib.Path)
    args = parser.parse_args(argv[1:])

    if args.gen_reference:
        gen_reference(*args.gen_reference,problemsizefile=args.problemsizefile)





























if __name__ == '__main__':
    retcode = main(argv=sys.argv)
    if retcode:
        exit(retcode)



