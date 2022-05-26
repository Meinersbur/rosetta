#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import pathlib
from pathlib import Path
import shutil


script = Path(sys.argv[0]).absolute()
thisscript = Path(__file__)

sys.path.insert(0,str( (thisscript.parent / 'rosetta' /  'lib').absolute() ))
#print(sys.path)
from rosetta import *
#import rosetta
#print(dir(rosetta))
#print(rosetta.__path__)


def main(argv):
    parser = argparse.ArgumentParser(description="Benchmark configure, build, execute & evaluate", allow_abbrev=False)
    add_boolean_argument(parser, 'clean', default=False, help="Start from scratch")
    add_boolean_argument(parser, 'configure', default=True, help="Enable configure (CMake) step")
    add_boolean_argument(parser, 'build', default=True, help="Enable build step")
    add_boolean_argument(parser, 'run', default=True, help="Enable run step")
    add_boolean_argument(parser, 'evaluate', default=True, help="Enable run step")
    parser.add_argument('--verbose', '-v', action='count')
    args = parser.parse_args(argv[1:])

    verbose = args.verbose

    def print_verbose(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    def invoke_verbose(*args, **kwargs):
        if verbose :
            invoke.diag(*args, **kwargs)
        else:
            invoke.run(*args, **kwargs)

    srcdir = script.parent
    builddir = srcdir / 'build'

    if builddir.exists():
        if args.clean:
            print_verbose("Cleaning previous run")
            shutil.rmtree(builddir)
        else:
            print_verbose("Reusing existing build")

    builddir.mkdir(exist_ok=True)

    # TODO: recognize "module" system
    # TODO: Recognize some famous machines (Summit, Aurora, Frontier, ...)

    libbuilddir = builddir/'rosetta-build'
    if (libbuilddir / 'build.ninja').exists():
        print_verbose("Already configured (Ninja will reconfigure when necessary automatically)")
    else:
        libbuilddir.mkdir(exist_ok=True)
        # TODO: Support other generators as well
        invoke_verbose('cmake', srcdir, '-GNinja Multi-Config', '-DCMAKE_CROSS_CONFIGS=all', cwd=libbuilddir)

    if args.build and not args.run:
        invoke_verbose('ninja', cwd=libbuilddir)

    if args.run:
        invoke_verbose('cmake', '--build', '.',  '--config','Release', '--target','run', cwd=libbuilddir)
    



if __name__ == '__main__':
    retcode = main(argv=sys.argv)
    if retcode:
        exit(retcode)


