#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Same as benchmark.py, but no need to launch a shell first
if __name__ == '__main__':
    import sys
    import pathlib

    scriptdir = pathlib.Path(__file__).parent
    sys.path.insert(0, str(scriptdir / 'rosetta' / 'src'))

    from rosetta.scripts.rosetta import main

    retcode = main(argv=sys.argv, rootdir=scriptdir)
    if retcode:
        exit(retcode)
