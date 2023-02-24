# -*- coding: utf-8 -*-

import os
import sys


print("test/__init__.py")



sys.path.insert(0,  os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
