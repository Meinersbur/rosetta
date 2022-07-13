#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from cmath import exp
from itertools import count
import sys
import argparse
import subprocess
import pathlib
import os
import datetime
import json
import xml.etree.ElementTree as et
import colorama  
import math
import argparse
from collections import defaultdict
import invoke
import io
from support import *

# Not included batteries
import cwcwidth
#import tqdm # progress meter


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
    def __init__(self, s, align=None):
        self.s = s
        self.align=align

    def __add__(self, other):
        return str_concat(self,other)

    def normalize(self):
        # StrAlign cannot be nested
        return self




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





class Table:
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
        matrix=[]
        nrows = len(self.rows)
        ncols = len(self.columns)

        collen = []
        colleft = []
        titles = []
        for i,name in enumerate(self.columns):
            vals = [r.get(name) for r in self.rows] 
            strs = []

            title = self.column_titles.get(name) or name
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
            matrix.append(strs)


        # Printing...
        def centering(s,collen):
            printlen = printlength(s)
            half = (collen - printlen)//2
            return ' '*half + consolestr(s) + ' ' *(collen - printlen - half)
        def raggedright(s,collen):
            printlen = printlength(s)
            return consolestr(s)  + ' ' * (collen - printlen)
        def aligned(s,maxlen,maxleft,alignpos):
            printlen = printlength(s)
            indent = maxleft - alignpos
            cs = consolestr(s)
            return ' ' * indent + cs + ' '*(maxlen - printlen - indent)
        def linesep():
           print(*(colorama.Style.DIM + '-'*collen[i] + colorama.Style.RESET_ALL for i in range(ncols))) 
        print()
        #linesep()
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
                    cs = consolestr(s.s)
                    left = cs[:s.align]
                    right = cs[s.align:] 
                    return f"{left:>{maxleft}}{right:<{maxright}}"

               
                # Left align by default
                return raggedright(s,maxlen)

            print(*(colval(i,name) for i,name in enumerate(self.columns)))
        #linesep()




# Summary statistics
# Don't trust yet, have to check correctness
class Statistic:
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
        if not mean :
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



class BenchResult:
    def __init__(self,bench, name:str, count:int, durations, maxrss, cold_count, peak_alloc):
        self.bench=bench
        self.name=name
        self.count=count
        #self.wtime=wtime
        #self.utime=utime
        #self.ktime=ktime
        #self.acceltime=acceltime
        self.durations=durations
        self.maxrss=maxrss
        self.cold_count = cold_count
        self.peak_alloc = peak_alloc


# TODO: enough prevision for nanoseconds?
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


def do_run(bench,args) :
    exe = bench.exepath 

    start = datetime.datetime.now()
    print ("Executing", shjoin( [exe] + args))
    p = subprocess.Popen([exe] + args ,stdout=subprocess.PIPE,universal_newlines=True)
    stdout = p.stdout.read()
    unused_pid, exitcode, ru = os.wait4(p.pid, 0)
    stop = datetime.datetime.now()

    wtime = max(stop - start,datetime.timedelta(0))
    utime = ru.ru_utime
    stime = ru.ru_stime
    maxrss = ru.ru_maxrss * 1024

    benchmarks = et.fromstring(stdout)

    count = 0
    for benchmark in benchmarks:
        name = benchmark.attrib['name']
        n = benchmark.attrib['n']
        cold_count = benchmark.attrib['cold_iterations']
        peak_alloc = benchmark.attrib['peak_alloc']
        count = len( benchmark)

        time_per_key = defaultdict(lambda :  [])
        for b in benchmark :
            for k, v in b.attrib.items():
                time_per_key[k] .append(parse_time(v))

        stat_per_key = {}
        for k,data in time_per_key.items():
            stat_per_key[k] = statistic(data)

        yield BenchResult(bench=bench, name=name, count=count,durations=stat_per_key, maxrss=maxrss,cold_count=cold_count,peak_alloc=peak_alloc) 



def run_gbench(bench,problemsizefile):
    yield from do_run(bench=bench, args=[f'--problemsizefile={problemsizefile}'])
        


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


def run_verify(problemsizefile):
    if not problemsizefile:
        die("Problemsizes required")  
    if not problemsizefile.is_file():
        # TODO: Embed default sizes
        die(f"Problemsize file {problemsizefile} does not exist.",file=sys.stderr)

    for e in benchmarks:
            exepath = e.exepath
            refpath = e.refpath
            #print(f"{exepath=}")

            args = [exepath, f'--verify', f'--problemsizefile={problemsizefile}']
            p = invoke.call(*args, return_stdout=True, print_command=True)
            data = p.stdout 
            #print(f"{data=}")

            with refpath.open() as f:
                refdata = f.read()
                if refdata != data:
                    # TODO: allow floating-point differences
                    print(f"Output different from reference for {e.target}")
                    print("Output   ", data)
                    print("Reference", refdata)
                    exit (1)


def get_problemsizefile(srcdir, problemsizefile):
    if not problemsizefile:
        if not srcdir:
            die("Problemsizefile must be defined")
        return mkpath(srcdir) / 'benchmarks' / 'medium.problemsize.ini'
    if not problemsizefile.is_file():
        # TODO: Embed default sizes
        die(f"Problemsize file {problemsizefile} does not exist.",file=sys.stderr)

    


def run_bench(problemsizefile=None, srcdir=None):
    problemsizefile = get_problemsizefile(srcdir, problemsizefile)

    results = []
    for e in benchmarks:
        results += list(run_gbench(e,problemsizefile=problemsizefile))

    stats_per_key = defaultdict(lambda :  [])
    for r in results:
        for k,stat in r.durations.items():
            stats_per_key[k] .append(stat)

    summary_per_key = {} # mean of means
    for k,data in stats_per_key.items():
        summary_per_key[k] = statistic(d.mean for d in data)


    table = Table()
    def path_formatter(v:pathlib.Path):
        if v is None:
            return None
        return StrColor(pathlib.Path(v).name,colorama.Fore.GREEN)
    def count_formatter(v:int):
        s = str(v)
        return StrAlign( StrColor(str(v),colorama.Fore.BLUE), printlength(s))
    def ppm_formatter(s:str):
        return getPPMDisplayStr(s)
    def duration_formatter(best,worst):
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

    table.add_column('program', title=StrColor("Benchmark", colorama.Fore.BWHITE),  formatter=path_formatter)
    table.add_column('ppm', title="PPM",formatter=ppm_formatter)
    table.add_column('config', title="Config")
    table.add_column('n', title=StrColor("Repeats", colorama.Style.BRIGHT),formatter=count_formatter)
    for k,summary in summary_per_key.items():
        table.add_column(k, title=StrColor( getMeasureDisplayStr(k), colorama.Style.BRIGHT),formatter=duration_formatter(summary.minimum,summary.maximum))


    for r in results:
        # TODO: acceltime doesn't always apply
        table.add_row(program=r.bench.target , ppm=r.bench.ppm, config=r.bench.config, n=r.count,**r.durations)

    
    table.print()


class Benchmark:
    def __init__(self,target,exepath,config,ppm,refpath):
        self.target=target
        self.exepath =exepath 
        self.config=config
        self.ppm = ppm
        self.refpath = refpath

    @property 
    def name(self):
        return self.target
        #return self.target.split(sep='.', maxsplit=1)[0]





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


# TODO: merge with run_gbench
# TODO: repeats for stability
def probe_bench(bench:Benchmark, limit_walltime, limit_rss, limit_alloc):
    assert limit_walltime or limit_rss or limit_alloc, "at least one limit required"

    def is_too_large(result):
        if limit_walltime is not None and result.durations['walltime'].mean >= limit_walltime:
            return True
        if limit_rss is not None and  result.maxrss >= limit_rss:
             return True
        if limit_alloc is not None and result.peakalloc >= limit_alloc:
            return True
        return False

    # Find a rough ballpark
    lower_n = 1
    n = 1
    while True:
        [result] = do_run(bench,args=[f'-n{n}', '--repeats=1'] )
        if is_too_large(result):
            break

        lower_n = n
        n *= 2

    # Bisect between lower_n and n
    def func(n):
        [result] = do_run(bench,args=[f'-n{n}', '--repeats=1'] )
        if is_too_large(result):
            return -1
        return 1
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

    with problemsizefile.open('w+') as f:
        for line in problemsizecontent:
            print(line,file=f)




def runner_main():
    parser = argparse.ArgumentParser(description="Benchmark runner", allow_abbrev=False)
    parser.add_argument('--problemsizefile', type=pathlib.Path, help="Problem sizes to use (.ini file)")

    # Command
    parser.add_argument('--verify', action='store_true',  help="Write reference output file")

    # Command 
    parser.add_argument('--bench',  action='store_true',  help="Run benchmark")

    # Command
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--write-problemsizefile', type=pathlib.Path)
    parser.add_argument('--limit-walltime', type=parse_time)
    parser.add_argument('--limit-rss', type=parse_memsize)
    parser.add_argument('--limit-alloc', type=parse_memsize)
 

    args = parser.parse_args(sys.argv[1:])


    if args.probe:
        return run_probe(problemsizefile=args.write_problemsizefile, limit_walltime=args.limit_walltime, limit_rss=args.limit_rss, limit_alloc=args.limit_alloc)

    if args.verify:
       return run_verify(problemsizefile=args.problemsizefile)


    return run_bench(problemsizefile=args.problemsizefile)






resultsdir = None
def rosetta_config(resultsdir):
    # TODO: Check if already set and different
    resultsdir = mkpath(resultsdir)



benchmarks =[]
def register_benchmark(target,exepath,config,ppm,refpath):
    bench = Benchmark(target,exepath=mkpath(exepath), config=config,ppm=ppm,refpath=mkpath(refpath))
    benchmarks.append(bench)




def load_register_file(filename):
    import importlib
    filename = str(filename)
    spec = importlib.util.spec_from_file_location(filename, filename)
    module =  importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)




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



