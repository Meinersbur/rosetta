#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


rosetta = __import__("rosetta.scripts.gen-stage2")
if __name__ == '__main__':
    retcode = getattr(rosetta.scripts, 'gen-stage2').main()
    if retcode:
        exit(retcode)
