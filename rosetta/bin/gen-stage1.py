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



print(pathlib.Path(__file__).parent.absolute() / 'lib')
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute() / 'lib'))
from support import *
import rosetta
runner = rosetta.runner

 


buildre = re.compile(r'^\s*//\s*BUILD\:(?P<script>.*)$')



def gen_benchtargets(outfile,problemsizefile,benchdir,builddir,configname):
    problemsizefile = runner.get_problemsizefile(problemsizefile=problemsizefile)
    config = configparser.ConfigParser()
    config.read(problemsizefile)

    buildfiles = []
    for path in benchdir.rglob('*'):
        #if path.name =='pointwise.cxx':
        if path.suffix.lower() in    {'.cxx', '.cpp', '.cu', '.build'}:
            buildfiles.append(path)

    benchs = []
    configdepfiles = []

    for buildfile in buildfiles:
        with buildfile.open() as f:
            script = []
            while line:= f.readline():
                if m := buildre.match(line):
                    s = m.group('script')
                    script.append(s)
            if script:
                configdepfiles.append(buildfile)
                script  = '\n'.join(script)
                spec = importlib.util.spec_from_loader('rosetta.build', loader=None, origin=buildfile)
                module =  importlib.util.module_from_spec(spec)
                globals = module.__dict__
                def add_benchmark(sources=None,basename=None,ppm=None):
                    # Guess sources (same file)
                    if not sources:
                        sources = [buildfile]
                    firstsource = mkpath(sources[0])

                    # Generate a target name
                    rel = firstsource.parent.parent.relative_to(benchdir)

                    # Guess basename
                    if not basename:
                        basename = firstsource.stem
                        basename = basename.removesuffix('.omp_parallel')
                        basename = '.'.join(list(rel.parts) + [basename])  
                    
                    target =  basename  + '.' + ppm   


                    bench = runner.Benchmark(basename=basename,target=target,exepath=None,ppm=ppm,configname=configname,config=None,sources=sources)
                    benchs.append(bench)
 
                globals['add_benchmark'] = add_benchmark
                globals['serial'] = 'serial'
                globals['__file__'] = buildfile
                exec(script, module.__dict__)


    with outfile.open("w+") as out:
        if configdepfiles:
            print("set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS",file=out)
            print('"' + ';'.join(pyescape(s) for s in configdepfiles) + '")',file=out)
            print(file=out)
        
        for bench in benchs:
            print(f"add_benchmark_serial({bench.basename}",file=out)
            print("SOURCES", *bench.sources  ,file=out)
            print(")",file=out)






def main():
    print("argv", sys.argv)
    parser = argparse.ArgumentParser(description="Generate CMakeLists.txt include file", allow_abbrev=False)
    parser.add_argument('--output', type=pathlib.Path)
    parser.add_argument('--problemsizefile', type=pathlib.Path)
    parser.add_argument('--benchdir', type=pathlib.Path)
    parser.add_argument('--builddir', type=pathlib.Path)
    parser.add_argument('--configname')
    args = parser.parse_args()

    gen_benchtargets(outfile=args.output, problemsizefile=args.problemsizefile, benchdir=args.benchdir, builddir=args.builddir,configname=args.configname)

 


if __name__ == '__main__':
    retcode = main()
    if retcode:
        exit(retcode)
