# -*- coding: utf-8 -*-

import tempfile
import contextlib
import math

from .util.support import mkpath

# TODO: enough precision for nanoseconds?
# TODO: Use alternative duration class


def parse_time(s: str):
    if s.endswith("ns"):
        return float(s[:-2]) / 1000000000
    if s.endswith("us") or s.endswith("µs"):
        return float(s[:-2]) / 1000000
    if s.endswith("ms"):
        return float(s[:-2]) / 1000
    if s.endswith("s"):
        return float(s[:-1])
    if s.endswith("m"):
        return float(s[:-1]) * 60
    if s.endswith("h"):
        return float(s[:-1]) * 60 * 60
    raise Exception("Don't know the duration unit")


# TODO: Recognize Kibibytes
def parse_memsize(s: str):
    if s.endswith("K"):
        return math.ceil(float(s[:-1]) * 1024)
    if s.endswith("M"):
        return math.ceil(float(s[:-1]) * 1024 * 1024)
    if s.endswith("G"):
        return math.ceil(float(s[:-1]) * 1024 * 1024 * 1024)
    return int(s)


mytempdir = None
globalctxmgr = contextlib.ExitStack()


def request_tempdir(subdir=None):
    global mytempdir
    global tempdirhandle
    if not mytempdir:
        # TODO: Option to not delete / keep in current directory
        tempdirhandle = tempfile.TemporaryDirectory(prefix=f'rosetta-')
        mytempdir = mkpath(globalctxmgr.enter_context(tempdirhandle))

        def clear_tempdirhandle():
            # Ensure we don't try to reuse the same (deleted) tempdir
            global mytempdir
            mytempdir = None
        globalctxmgr.callback(clear_tempdirhandle)

    #print(f"Tempdir is: {mytempdir}")
    if subdir:
        subtmpdir = mytempdir / subdir
        subtmpdir.mkdir(exist_ok=True)
        return subtmpdir

    return mytempdir


def request_tempfilename(prefix=None, suffix=None, subdir=None):
    tmpdir = request_tempdir(subdir=subdir)
    candidate = tmpdir / f'{prefix}{suffix}'
    i = 0
    while candidate.exists():
        candidate = tmpdir / f'{prefix}-{i}{suffix}'
        i += 1

    return candidate
