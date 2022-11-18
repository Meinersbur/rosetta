#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import configparser
import importlib
import pathlib
import types
import sys
import re
import importlib.util



#print(pathlib.Path(__file__).parent.absolute() / 'lib')
#sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute() / 'lib'))
from rosetta.util.support import *
import rosetta
runner = rosetta.runner
generator = rosetta.generator
 


buildre = re.compile(r'^\s*//\s*BUILD\:(?P<script>.*)$')
preparere = re.compile(r'^\s*//\s*PREPARE\:(?P<script>.*)$')



def compute_global_indent(slist):
    gindent = None
    for line in slist:
        if not  line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if gindent is None or indent < gindent:
            gindent = indent
    return gindent or 0
    
def unindent(slist, amount):
 for line in slist:
    yield line[amount:]

def global_unintent(slist):
    gindent = compute_global_indent(slist)
    yield from unindent(slist,gindent)


def gen_benchtargets(outfile,problemsizefile,benchdir,builddir,configname,filter=None):
    problemsizefile = runner.get_problemsizefile(problemsizefile=problemsizefile)
    config = configparser.ConfigParser()
    config.read(problemsizefile)

    buildfiles = []
    for path in benchdir.rglob('*'):
        #if path.name =='pointwise.cxx':
        if not path.suffix.lower() in    {'.cxx', '.cu', '.build'}:
            continue
        if filter and not filter in path.name:
            continue
        buildfiles.append(path)

    benchs = []
    configdepfiles = []

    for buildfile in buildfiles:
        with buildfile.open() as f:
            script = []
            while True:
                line= f.readline()
                if not line:
                    break
                m = buildre.match(line)
                if m:
                    s = m.group('script')
                    script.append(s)
            if script:
                #print("Processing:", buildfile)
                configdepfiles.append(buildfile)
                script = global_unintent(script)
                script  = '\n'.join(script)
                spec = importlib.util.spec_from_loader('rosetta.build', loader=None, origin=buildfile)
                module =  importlib.util.module_from_spec(spec)
                globals = module.__dict__
                scriptdir = buildfile.parent
                relbuildfile = buildfile.relative_to(scriptdir)
                def add_benchmark(sources=None,basename=None,ppm=None):
                    if  sources is not None:
                        mysources = []
                        for s in sources:
                            mysources.append(scriptdir / mkpath(s))
                    else:
                        # Guess sources (same file)
                        mysources = [buildfile]
                    firstsource = mkpath(mysources[0])

                    # Generate a target name
                    rel = firstsource.parent.parent.relative_to(benchdir)

                    # Guess basename
                    if not basename:
                        basename = firstsource.stem
                        basename = removesuffix(basename,'.omp_parallel')
                        basename = removesuffix(basename,'.omp_task')
                        basename = removesuffix(basename,'.omp_target')
                        basename = '.'.join(list(rel.parts) + [basename])  
                    
                    target =  basename  + '.' + ppm   
                    pbsize = config.getint(basename, 'n')

                    bench = runner.Benchmark(basename=basename,target=target,exepath=None,ppm=ppm,configname=configname,buildtype=None,sources=mysources,pbsize=pbsize)
                    benchs.append(bench)
 

                globals['add_benchmark'] = add_benchmark
                globals['serial'] = 'serial'
                globals['cuda'] = 'cuda'
                globals['omp_parallel'] = 'omp_parallel'
                globals['omp_task'] = 'omp_task'
                globals['omp_target'] = 'omp_target'
                globals['__file__'] = relbuildfile
                #print("Executing:")
                #print(script)
                exec(script, module.__dict__)



    with outfile.open("w+") as out:
        if configdepfiles:
            print("set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS",file=out)
            print('"' + ';'.join(pyescape(s) for s in configdepfiles) + '")',file=out)
            print(file=out)
        
        for bench in benchs:
            if bench.ppm == 'serial':
                print(f"add_benchmark_serial({bench.basename}",file=out)
            elif bench.ppm == 'omp_parallel':
                print(f"add_benchmark_openmp_parallel({bench.basename}",file=out)
            elif bench.ppm == 'omp_task':
                print(f"add_benchmark_openmp_task({bench.basename}",file=out)
            elif bench.ppm == 'omp_target':
                print(f"add_benchmark_openmp_target({bench.basename}",file=out)
            elif bench.ppm == 'cuda':
                print(f"add_benchmark_cuda({bench.basename}",file=out)
            else:
                die("Unhandled ppm")
            print("    PBSIZE", bench. pbsize ,file=out)
            print("    SOURCES", *(bs.as_posix() for bs in bench.sources)  ,file=out)
            print("  )",file=out)






def main():
    #print("stage1 argv", sys.argv)
    parser = argparse.ArgumentParser(description="Generate CMakeLists.txt include file", allow_abbrev=False)
    parser.add_argument('--output', type=pathlib.Path)
    parser.add_argument('--problemsizefile', type=pathlib.Path)
    parser.add_argument('--benchdir', type=pathlib.Path)
    parser.add_argument('--builddir', type=pathlib.Path)
    parser.add_argument('--configname')
    parser.add_argument('--filter', help="Only look into filenames that contain this substring") # TODO: More extensive filter mechanisms
    args = parser.parse_args()

    gen_benchtargets(outfile=args.output, problemsizefile=args.problemsizefile, benchdir=args.benchdir, builddir=args.builddir,configname=args.configname,filter=args.filter)

 


if __name__ == '__main__':
    retcode = main()
    if retcode:
        exit(retcode)
