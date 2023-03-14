#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from rosetta import *
from rosetta.util.support import *
from rosetta.util.cmdtool import *
from rosetta.common import *
from rosetta.driver import driver_main, DriverMode


def main(argv=sys.argv, rootdir=None):
    srcdir = os.path.join(__file__, '..',  '..', '..', '..', 'benchmarks')
    driver_main(argv=argv, mode=DriverMode.MANAGEDBUILDDIR,
                rootdir=rootdir, srcdir=srcdir)


if __name__ == '__main__':
    retcode = main(argv=sys.argv)
    if retcode:
        exit(retcode)
