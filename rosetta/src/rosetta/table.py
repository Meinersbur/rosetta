# -*- coding: utf-8 -*-

import cwcwidth
import importlib.util
import importlib
import contextlib
import typing
import configparser
import io
from collections import defaultdict
import math
import colorama
import xml.etree.ElementTree as et
from typing import Iterable
import json
import datetime
import os
import pathlib
import subprocess
import argparse
import sys
from itertools import count
from .util.cmdtool import *
from .util.orderedset import OrderedSet
from .util.support import *
from .util import invoke


# FIXME: Hack
colorama.Fore.BWHITE = colorama.ansi.code_to_chars(97)


class StrConcat:  # Rename: Twine
    def __init__(self, args):
        self.args = list(args)

    def __add__(self, other):
        common = self.args + [other]
        return str_concat(*common)

    def printlength(self):
        return sum(printlength(a) for a in self.args)

    def normalize(self):
        # Optional: collapse nested StrConact
        # Optional: concat neighboring string

        # hoist StrAlign to topmost
        for i, a in enumerate(self.args):
            a = normalize(a)
            if isinstance(a, StrAlign):
                prefixlen = sum(printlength(a) for a in self.args[:i])
                return StrAlign(StrConcat(self.args[:i] + [a.s] + self.args[i + 1:]), prefixlen + a.align)
        return self

    def consolestr(self):
        return ''.join(consolestr(a) for a in self.args)


class StrColor:
    def __init__(self, s, style):
        self.s = s
        self.style = style

    def __add__(self, other):
        return str_concat(self, other)

    def printlength(self):
        return printlength(self.s)

    def normalize(self):
        a = normalize(self.s)
        if isinstance(a, StrAlign):
            return StrAlign(StrColor(a.s, self.style), a.align)
        if a is self.s:
            return self
        return StrColor(a, self.style)

    def consolestr(self):
        from colorama import Fore, Back, Style
        if self.style in {Fore.BLACK, Fore.RED, Fore.GREEN, Fore.YELLOW,
                          Fore. BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE, Fore.BWHITE}:
            reset = Fore.RESET
        elif self.style in {Back.BLACK, Back.RED, Back.GREEN, Back.YELLOW, Back. BLUE, Back.MAGENTA, Back.CYAN, Back.WHITE}:
            reset = Back.RESET
        elif self.style in {Style.DIM, Style.NORMAL, Style.BRIGHT}:
            reset = Style.RESET_ALL
        else:
            reset = ''
        return self.style + consolestr(self.s) + reset


class StrAlign:
    LEFT = NamedSentinel('LEFT')
    CENTER = NamedSentinel('CENTER')
    RIGHT = NamedSentinel('RIGHT')

    # TODO: swap align/pos
    def __init__(self, s, align=None, pos=LEFT):
        self.s = s
        self.align = align
        self.pos = pos

    def __add__(self, other):
        return str_concat(self, other)

    def normalize(self):
        # StrAlign cannot be nested
        return self

    def printlength(self):
        return printlength(self.s)


def align_decimal(s):
    # FIXME: Don't align in scientific notation?
    pos = s.find('.')
    if pos >= 0:
        return StrAlign(s, pos)
    return StrAlign(s, printlength(s))





def str_concat(*args):
    return StrConcat(args=args)


def consolestr(s):
    if isinstance(s, str):
        return s
    return s.consolestr()


def printlength(s):
    if isinstance(s, str):
        result = cwcwidth.wcswidth(s)
        if result != len(s):
            pass
        return result
    return s.printlength()


def normalize(s):
    if isinstance(s, str):
        return s
    return s.normalize()


def default_formatter(v):
    return str(v)





# TODO: Support subcolumns
class Table:
    def __init__(self):
        self.columns = []
        self.supercolumns = dict()
        self.column_titles = dict()
        self.column_formatters = dict()
        self.rows = []

    def add_column(self, name, title=None, formatter=None):
        assert name not in self.columns, "Column names must be unique"
        self.columns.append(name)
        if title is not None:
            self.column_titles[name] = title
        if formatter is not None:
            self.column_formatters[name] = formatter

    def make_supercolumn(self, name, subcolumns):
        assert subcolumns, "Supercolumn exists of at least one column"
        assert name in self.columns, "add column first before making it a supercolumn"
        assert name not in self.supercolumns, "already is a supercolumn"
        self.supercolumns[name] = subcolumns

    def add_row(self, **kwargs):
        self.rows .append(kwargs)

    def print(self):
        matrix = []
        nrows = len(self.rows)
        ncols = len(self.columns)

        colname_to_idx = dict()
        for i, name in enumerate(self.columns):
            colname_to_idx[name] = i

        name_to_leafidx = dict()
        leafidx_to_name = dict()  # TODO: array
        for i, name in enumerate(name for name in self.columns if name not in self.supercolumns):
            name_to_leafidx[name] = i
            leafidx_to_name[i] = name

        # Determine columns and their max width
        collen = []
        colleft = []
        colright = []
        titles = []
        for i, name in enumerate(self.columns):
            vals = [r.get(name) for r in self.rows]
            strs = []

            # TODO: Handle title just like another row
            title = self.column_titles.get(name) or name  # TODO: Allow empty titles for supercolumns
            formatter = self.column_formatters.get(name) or default_formatter
            maxlen = printlength(title)
            titles.append(title)
            maxleft = 0
            maxright = 0

            for v in vals:
                if v is None:
                    strs.append(None)
                else:
                    s = formatter(v)  # TODO: support embedded newlines
                    s = normalize(s)
                    if isinstance(s, StrAlign):
                        l = printlength(s.s)
                        left = s.align
                        right = l - left
                        maxleft = max(maxleft, left)
                        maxright = max(maxright, right)
                    else:
                        l = printlength(s)
                        maxlen = max(maxlen, l)
                    strs.append(s)

            maxlen = max(maxlen, maxleft + maxright)
            collen.append(maxlen)
            colleft.append(maxleft)
            colright.append(maxright)
            matrix.append(strs)

        # Adapt for supercolumns
        # TODO: supercolumns might be hierarchical, so order is relevant TODO:
        # determine the range of leaf columns in advance
        for supercol, subcols in self.supercolumns.items():
            subcollen = sum(collen[colname_to_idx.get(s)] for s in subcols)

            # print() inserts one space between items
            subcollen += len(subcols) - 1
            supercollen = collen[colname_to_idx.get(supercol)]
            if subcollen < supercollen:
                # supercolumn is wider than subcolumns: divide additional space evenly between subcolumns
                overhang = supercollen - subcollen
                for i, subcol in enumerate(subcols):
                    addlen = ((i + 1) * overhang + len(subcols) // 2) // len(subcols) - \
                        (i * overhang + len(subcols) // 2) // len(subcols)
                    collen[colname_to_idx.get(subcol)] += addlen
            elif subcollen > supercollen:
                # subcolumns are wider than supercolumn: extend supercolumn
                collen[colname_to_idx.get(supercol)] = subcollen

        # Printing...

        def centering(s, collen):
            printlen = printlength(s)
            half = (collen - printlen) // 2
            return ' ' * half + consolestr(s) + ' ' * (collen - printlen - half)

        def raggedright(s, collen):
            printlen = printlength(s)
            return consolestr(s) + ' ' * (collen - printlen)

        def aligned(s, maxlen, maxleft, alignpos):
            if alignpos is None:
                return raggedright(s, maxlen)
            else:
                printlen = printlength(s)
                indent = maxleft - alignpos
                cs = consolestr(s)
                return ' ' * indent + cs + ' ' * (maxlen - printlen - indent)

        def linesep():
            print(*(colorama.Style.DIM + '-' * collen[colname_to_idx.get(colname)] +
                  colorama.Style.RESET_ALL for i, colname in leafidx_to_name.items()))

        def print_row(rowdata: dict):
            leafcolnum = len(name_to_leafidx)

            lines = [[' ' * collen[colname_to_idx.get(leafidx_to_name.get(j))] for j in range(0, leafcolnum)]]
            currow = [0] * leafcolnum

            def set_cells(celldata, supercol, cols):
                nonlocal lines, currow
                if not celldata:
                    return
                indices = [name_to_leafidx.get(c) for c in cols]
                start = min(indices)
                stop = max(indices)
                for i in range(start, stop + 1):
                    currow[i] += 1
                needlines = max(currow[cur] for cur in range(start, stop + 1))
                while len(lines) < needlines:
                    lines.append([" " * collen[colname_to_idx.get(leafidx_to_name.get(j))]
                                 for j in range(0, leafcolnum)])

                def colval(s, maxlen, maxleft, maxright):
                    #maxlen = collen[i]
                    #maxleft = colleft[i]

                    if isinstance(s, StrAlign):
                        if s.pos == StrAlign.LEFT:
                            return aligned(s.s, maxlen, maxleft, s.align)
                        elif s.pos == StrAlign.CENTER:
                            if s.align is None:
                                return centering(s.s, maxlen)
                            # TODO: check correctness
                            printlen = printlength(s)
                            rightindent = maxright - printlen - s.align
                            return centering(consolestr(s) + ' ' * rightindent)
                        elif s.pos == StrAlign.RIGHT:
                            if s.align is None:
                                return raggedright(s, maxlen)
                            # TODO: check correctness
                            printlen = printlength(s)
                            rightindent = maxright - printlen - s.align
                            leftindent = maxlen - rightindent - printlen
                            return ' ' * leftindent + consolestr(s) + ' ' * rightindent

                    # Left align by default
                    return raggedright(s, maxlen)

                totallen = sum(collen[colname_to_idx.get(leafidx_to_name.get(j))]
                               for j in range(start, stop + 1)) + stop - start
                totalleft = colleft[colname_to_idx.get(supercol)]
                totalright = colright[colname_to_idx.get(supercol)]
                lines[needlines - 1][start] = colval(celldata, totallen, totalleft, totalright)
                for i in range(start + 1, stop + 1):
                    lines[needlines - 1][i] = None

            for supercol, subcols in self.supercolumns.items():
                superdata = rowdata.get(supercol)
                set_cells(superdata, supercol, subcols)

            for i, colname in leafidx_to_name.items():
                celldata = rowdata.get(colname)
                set_cells(celldata, colname, [colname])

            for line in lines:
                print(*(l for l in line if l is not None))

        print()
        print_row(self.column_titles)
        linesep()

        for j in range(nrows):
            print_row({colname: matrix[i][j] for colname, i in colname_to_idx.items()})

        return
        print()
        print(*(centering(titles[i], collen[i]) for i in range(ncols)))
        linesep()

        for j in range(nrows):
            def colval(i, name):
                maxlen = collen[i]
                maxleft = colleft[i]
                s = matrix[i][j]

                if s is None:
                    return ' ' * maxlen
                if isinstance(s, StrAlign):
                    return aligned(s.s, maxlen, maxleft, s.align)

                # Left align by default
                return raggedright(s, maxlen)

            print(*(colval(i, name) for i, name in enumerate(self.columns)))
        # linesep()
