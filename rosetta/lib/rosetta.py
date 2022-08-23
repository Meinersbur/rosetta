#! /usr/bin/env python3
# -*- coding: utf-8 -*-

print("rosetta.py executed")


# TODO: Clenaup file separation
from cmdtool import *
import invoke
from runner import runner_main,register_benchmark,load_register_file,rosetta_config
import runner
from orderedset import OrderedSet
from runner import Benchmark
import generator
