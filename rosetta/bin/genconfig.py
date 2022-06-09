#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import sys
print(pathlib.Path(__file__).parent.absolute() / 'lib')
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute() / 'lib'))
#sys.path.insert(0, '/home/meinersbur/src/rosetta/rosetta/lib')


from rosetta import *
import argparse
import pathlib 
import configparser



def gen_config(output, benchname,problemsizefile):
    config = configparser.ConfigParser()
    config.read(problemsizefile)
    n = config.getint(benchname, 'n')

    content = f"""#include <cstdint>
    
const char *bench_name = "{benchname}";
int64_t bench_default_problemsize = {n};
"""

    with output.open('w+') as f:
        f.write(content)



def main() :
    parser = argparse.ArgumentParser(description="Benchmark configuration file generator", allow_abbrev=False)
    parser.add_argument('--output', type=pathlib.Path)
    parser.add_argument('--benchname')
    parser.add_argument('--problemsizefile',  type=pathlib.Path)
    args = parser.parse_args()

    gen_config(output = args.output, benchname = args.benchname, problemsizefile =args. problemsizefile)



if __name__ == '__main__':
    retcode = main()
    if retcode:
        exit(retcode)



