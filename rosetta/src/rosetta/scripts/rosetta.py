#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from rosetta.util import support
import sys
from collections import defaultdict
import sys
import argparse
import pathlib
from pathlib import Path
import shutil
import re
import importlib
import rosetta
from rosetta import *
import rosetta.runner as runner
from rosetta.util.support import *
from rosetta.util.cmdtool import *
from rosetta.util.orderedset import OrderedSet
import rosetta.util.invoke as invoke
from rosetta.evaluator import subcommand_evaluate
from rosetta.runner import subcommand_run
from rosetta.common import *
from rosetta.driver import driver_main,DriverMode



def main(argv=sys.argv,rootdir=None):
    srcdir = os.path.join(__file__ , '..',  '..', '..', '..', 'benchmarks')
    driver_main(argv=argv,mode=DriverMode.MANAGEDBUILDDIR,rootdir=rootdir,srcdir=srcdir)


if __name__ == '__main__':
    retcode = main(argv=sys.argv)
    if retcode:
        exit(retcode)
