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



def main():
    print("argv", sys.argv)
    parser = argparse.ArgumentParser(description="Generate make-time files", allow_abbrev=False)
    parser.add_argument('--output', type=pathlib.Path)
    parser.add_argument('--problemsizefile', type=pathlib.Path)
    parser.add_argument('--benchdir', type=pathlib.Path)
    parser.add_argument('--builddir', type=pathlib.Path)
    parser.add_argument('--configname')
    args = parser.parse_args()

    #gen_refsizeinclude(output=args.output, problemsizefile=args.problemsizefile)
    gen_benchtargets(outfile=args.output, problemsizefile=args.problemsizefile, benchdir=args.benchdir, builddir=args.builddir,configname=args.configname)

 

if __name__ == '__main__':
    retcode = main()
    if retcode:
        exit(retcode)
