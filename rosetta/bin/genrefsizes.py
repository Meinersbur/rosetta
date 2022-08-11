#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import configparser
import importlib
import pathlib
import sys
import re
print(pathlib.Path(__file__).parent.absolute() / 'lib')
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute() / 'lib'))

from support import *
import rosetta
runner = rosetta.runner


def gen_refsizeinclude(output,problemsizefile):
    #print("problemsizefile", problemsizefile)
    problemsizefile = runner.get_problemsizefile(problemsizefile=problemsizefile)
    #print("problemsizefile", problemsizefile)

    config = configparser.ConfigParser()
    config.read(problemsizefile)
    
    #result = f"message({pystr(problemsizefile)})\n"
    result = ""
    for secname in config.sections():
        n = config.getint(section=secname,option="n")
        print(secname, n)
        result += f"rosetta_add_reference(\"{secname}\" {n})\n"

    createfile(output, result)
  
 


buildre = re.compile(r'^\s*//\*')


def gen_benchtargets(outfile,problemsizefile,benchdir,builddir):
    problemsizefile = runner.get_problemsizefile(problemsizefile=problemsizefile)
    config = configparser.ConfigParser()
    config.read(problemsizefile)

    buildfiles = []
    for path in benchdir.rglob('*'):
        if path.suffix.lower() in {'.cxx', '.cu', '.build'}:
            buildfiles.append(path)

    for buildfile in buildfiles:
        with buildfile.open() as f:
            while line:= f.readline():
                s = line.split


    filename = str(filename)
    spec = importlib.util.spec_from_file_location(filename, filename)
    module =  importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)



def main():
    print("argv", sys.argv)
    parser = argparse.ArgumentParser(description="Generate CMakeLists.txt include file", allow_abbrev=False)
    parser.add_argument('--output', type=pathlib.Path)
    parser.add_argument('--problemsizefile', type=pathlib.Path)
    parser.add_argument('--benchdir', type=pathlib.Path)
    parser.add_argument('--buildir', type=pathlib.Path)
    args = parser.parse_args()

    gen_refsizeinclude(output=args.output, problemsizefile=args.problemsizefile)
    #gen_benchtargets(output=args.output, problemsizefile=args.problemsizefile, benchdir=args.benchdir, builddir=args.builddir)

 


if __name__ == '__main__':
    retcode = main()
    if retcode:
        exit(retcode)
