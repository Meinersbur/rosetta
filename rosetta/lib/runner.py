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
import colorama  
import cwcwidth

# FIXME: Hack
colorama.Fore.BWHITE = colorama.ansi.code_to_chars(97)


class StrConcat:
    def __init__(self, args):
        self.args = list(args)

    def __add__(self, other):
        common = self.args + [other]
        return str_concat(self,*common)

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



class BenchVariants:
    def __init__(self, default_size, serial=None, cuda=None):
        None



class BenchResult:
    def __init__(self,name:str, count:int, wtime , utime ,ktime , acceltime,maxrss):
        self.name=name
        self.count=count
        self.wtime=wtime
        self.utime=utime
        self.ktime=ktime
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
            usertime = float(it.attrib['usertime'])
            usersum += usertime
            kerneltime = float(it.attrib['kerneltime'])
            kernelsum += kerneltime
            if 'acceltime' in it.attrib:
                if acceltimesum is  None:
                    acceltimesum = 0
                acceltime  = float(it.attrib['acceltime'])
                acceltimesum += acceltime
        yield BenchResult(name=name, count=count,wtime=walltime/count,utime=usersum/count,ktime=kernelsum/count,acceltime=None if acceltime is None else acceltimesum/count, maxrss=maxrss) 
        


def align_decimal(s):
    pos = s.find('.')
    if pos >= 0:
        return StrAlign(s, pos)
    return StrAlign(s, len(s))



def run_benchs(config:str=None,serial=[],cuda=[]):
    results = []
    for e in serial:
        results += list(run_gbench(exe=e))

    for e in cuda:
        results += list(run_gbench(exe=e))

    table = Table()
    def path_formatter(v:pathlib.Path):
        if v is None:
            return None
        return  StrColor( pathlib.Path(v).name,colorama.Fore.GREEN)
    def duration_formatter(v):
        if v is None:
            return None
        if v >= 1:
            return align_decimal(f"{v:.2}") +StrColor( "s", colorama.Style.DIM)
        if v*1000 >= 1:
            return align_decimal(f"{v*1000:.2}") + StrColor("ms", colorama.Style.DIM)
        if v*1000*1000 >= 1:
            return align_decimal(f"{v*1000*1000:.2}") + StrColor("µs", colorama.Style.DIM)
        return align_decimal(f"{v*1000*1000*1000:.2}") + StrColor( "ns", colorama.Style.DIM)

    table.add_column('program', title=StrColor("Benchmark", colorama.Fore.BWHITE),  formatter=path_formatter)
    table.add_column('n', title=StrColor( "n", colorama.Style.BRIGHT))
    table.add_column('wtime', title=StrColor( "Wall" , colorama.Style.BRIGHT),formatter=duration_formatter)
    table.add_column('utime', title=StrColor( "User" , colorama.Style.BRIGHT),formatter=duration_formatter)
    table.add_column('ktime', title=StrColor("Kernel" , colorama.Style.BRIGHT),formatter=duration_formatter)
    table.add_column('acceltime', title=StrColor("GPU" , colorama.Style.BRIGHT),formatter=duration_formatter)
    

    #print("Name: WallTime RealTime AccelTime MaxRSS")
    for r in results:
        table.add_row(program=r.name,n=r.count,wtime=r.wtime,utime=r.utime,ktime=r.ktime,acceltime=r.acceltime)
        #print(f"{r.name}: {r.wtime} {r.rtime} {r.acceltime} {r.maxrss}")

    
    table.print()



def main(argv):
    colorama.init()
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
