#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import configparser

import pathlib
import sys
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
    
    result = f"message({pystr(problemsizefile)})\n"
    for secname in config.sections():
        n = config.getint(section=secname,option="n")
        print(secname, n)
        result += f"rosetta_add_reference(\"{secname}\" {n})\n"

    createfile(output, result)
  
 

def main():
    print("argv", sys.argv)
    parser = argparse.ArgumentParser(description="Generate CMakeLists.txt include file", allow_abbrev=False)
    parser.add_argument('--output', type=pathlib.Path)
    parser.add_argument('--problemsizefile', type=pathlib.Path)
    args = parser.parse_args()

    gen_refsizeinclude(output=args.output, problemsizefile=args.problemsizefile)


 


if __name__ == '__main__':
    retcode = main()
    if retcode:
        exit(retcode)
