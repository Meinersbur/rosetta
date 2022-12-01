# -*- coding: utf-8 -*-


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
import html
import pathlib
import subprocess
import argparse
import sys
from itertools import count
import base64
import matplotlib.pyplot as plt # TODO: Only depend on matplotlib when really needed
import matplotlib.colors as mcolors
from cycler import cycler
from .util.cmdtool import *
from .util.orderedset import OrderedSet
from .util.support import *
from .util import invoke
from .table import *
from .stat import *
from .common import *




def subcommand_evaluate(parser,args,resultfiles,resultsdir):
    """
Evaluate a set of results. This can be from just executed benchmarks or reading from xml files.

Parameters
----------
parser : ArgumentParser
    ArgumentParser from argparse for adding arguments
args
    Parsed command line from argparse
resultfiles
    Subjects to evaluate
"""

    if parser:
        add_boolean_argument(parser, 'evaluate', default=None, help="Evaluate result")

        parser.add_argument('--boxplot', type=pathlib.Path, metavar="FILENAME", help="Save as boxplot to FILENAME")
        #parser.add_argument('--report', type=pathlib.Path, metavar="FILENAME",  help="Save a html reportfile")
        #add_boolean_argument(parser, 'report', default=None, help="Write rem result")

    if args:
        if args.evaluate:
            results = load_resultfiles(resultfiles)

            # Remove bogus entries
            # results = [r for r in results if   r.durations.get('walltime', statistic([]) ).count >= 1 ]

            #if len(builddirs) > 1:
            results_compare(results, compare_by="configname", compare_val=["walltime"])
            #else:
            #    evaluate(resultfiles)

            if args.boxplot:
                fig = results_boxplot(results)
                fig.savefig(fname=args.boxplot)
                fig.canvas.draw_idle()

            now = datetime.datetime.now() # TODO: Use runner.make_resultssubdir
            reportfile = resultsdir /  f"report_{now:%Y%m%d_%H%M}.html"

            # first_defined(args.report,resultsdir /  f"report_{now:%Y%m%d_%H%M}.html" )
            save_report(results,filename=reportfile)






def name_or_list(data):
    if not data:
        return None
    if isinstance(data, str):
        return data
    if not isinstance(data, Iterable):
        return data
    if len(data) == 1:
        return data[0]
    return data



#TODO: dataclass
class BenchResult:
    categorical_cols = ['program', 'ppm', 'buildtype', 'configname']
    numerical_cols = ['walltime']
    def __init__(self, name: str, ppm: str, buildtype: str, configname: str,
                 count: int, durations, maxrss=None, cold_count=None, peak_alloc=None):
        # self.bench=bench
        self.name = name
        self.ppm = ppm
        self.buildtype = buildtype
        self.configname = configname
        self.count = count
        # self.wtime=wtime
        # self.utime=utime
        # self.ktime=ktime
        # self.acceltime=acceltime
        self.durations = durations
        self.maxrss = maxrss
        self.cold_count = cold_count
        self.peak_alloc = peak_alloc


# TODO: Member of BenchResult
def get_column_data(result: BenchResult, colname: str):
    if result is None:
        return None
    if colname == "program":
        return result.name
    if colname == "ppm":
        return result.ppm
    if colname == "buildtype":
        return result.buildtype
    if colname == "configname":
        return first_defined(result.configname, "") #FIXME: "defaultbuild" is just placeholder
    if colname == "walltime":
        return result.durations.get("walltime")
    assert False, "TODO: Add to switch of use getattr"



def path_formatter(v: pathlib.Path):
    if v is None:
        return None
    return StrColor(pathlib.Path(v).name, colorama.Fore.GREEN)


def program_formatter(v: pathlib.Path):
    if v is None:
        return None
    return StrColor(v, colorama.Fore.GREEN)


def duration_formatter(best=None, worst=None):
    def formatter(s: Statistic):
        if s is None:
            return None
        v = s.mean
        d = s.relerr()

        def highlight_extremes(s):
            if best is not None and worst is not None and best < worst:
                if v <= best:
                    return StrColor(s, colorama.Fore.GREEN)
                if v >= worst:
                    return StrColor(s, colorama.Fore.RED)
            return s

        if d and d >= 0.0001:
            errstr = f"(±{d:.1%})"
            if d >= 0.02:
                errstr = StrColor(errstr, colorama.Fore.RED)
            errstr = str_concat(' ', errstr)
        else:
            errstr = ''

        if v >= 1:
            return highlight_extremes(align_decimal(f"{v:.2}")) + StrColor("s",
                                                                           colorama.Style.DIM) + (str_concat(' ', errstr) if errstr else '')
        if v * 1000 >= 1:
            return highlight_extremes(align_decimal(f"{v*1000:.2f}")) + StrColor("ms", colorama.Style.DIM) + errstr
        if v * 1000 * 1000 >= 1:
            return highlight_extremes(align_decimal(f"{v*1000*1000:.2f}")) + StrColor("µs", colorama.Style.DIM) + errstr
        return highlight_extremes(align_decimal(f"{v*1000*1000*1000:.2f}")) + \
            StrColor("ns", colorama.Style.DIM) + errstr
    return formatter














def results_boxplot(results, group_by=None, compare_by=None):
    r"""Produce a boxplot for benchmark results

    :param group_by:   Summerize all results that have the same value for these properties. No summerization if None.
    :param compare_by: Which property to compare side-by-side in a group of plots. Implicitly enables grouping.
    """


    if group_by or compare_by:
        if group_by is None:
            group_by = ["program", "ppm", "buildtype", "configname"]
        if compare_by:
            group_by.remove(compare_by)

        grouped_results, all_cmpvals, div_groups = grouping(results, compare_by='configname', group_by=group_by)
        groupdata = [[b.durations['walltime'].samples for b in group] for group in grouped_results]
    else:
        # Each result in its own group
        grouped_results = [[r] for r in results]
        div_groups = divergent_fields(["program", "ppm", "buildtype", "configname"], results)
        all_cmpvals = [""]

    def make_label(g: tuple):
        first = g[0]
        return ', '.join(get_column_data(first, k) for k in div_groups)
    labels = [make_label(g) for g in grouped_results]


    left = 1
    right = 0.5
    numgroups = len(grouped_results)
    benchs_per_group = len(all_cmpvals)
    barwidth = 0.3
    groupwidth = 0.2 + benchs_per_group * barwidth
    width = left + right + groupwidth * numgroups
    fig, ax = plt.subplots(figsize=(width, 10))
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [c['color'] for j, c in zip(range(benchs_per_group), prop_cycle)]  # TODO: Consider seaborn palettes

    fig.subplots_adjust(left=left / width, right=1 - right / width, top=0.95, bottom=0.25)

    for i, group in enumerate(grouped_results):
        benchs_this_group = len(group)
        for j, benchstat in enumerate(group):  # TODO: ensure grouped_results non-jagged so colors match
            data = benchstat.durations['walltime'].samples  # TODO: Allow other datum that walltime
            rel = (j - benchs_this_group / 2.0 + 0.5) * barwidth
            box = ax.boxplot(data, positions=[i * groupwidth + rel],
                             notch=True, showmeans=False, showfliers=True, sym='+',
                             widths=barwidth,
                             patch_artist=True,  # fill with color
                             )
            for b in box['boxes']:
                b.set_facecolor(colors[j])
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    for j, (c, label) in enumerate(zip(colors, all_cmpvals)):
        # Dummy item to add a legend handle; like seaborn does
        rect = plt.Rectangle([0, 0], 0, 0,
                             # linewidth=self.linewidth / 2,
                             # edgecolor=self.gray,
                             facecolor=c,
                             label=label)
        ax.add_patch(rect)

    # TODO: Compute conf_intervals consistently like the table, preferable using the student-t test.
    # x.grid(linestyle='--',axis='y')
    ax.set(
        axisbelow=True,  # Hide the grid behind plot objects
        xlabel='Benchmark',
        ylabel='Walltime [s]',
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xticks([groupwidth * i for i in range(len(labels))])
    ax.set_xticklabels(labels, rotation=20, ha="right", rotation_mode="anchor")

    plt.legend()

    # for label in ax.get_xticklabels(): # https://stackoverflow.com/a/43153984
    #    label.set_ha("right")
    #    label.set_rotation(45)
    return plt.gcf()






def load_resultfiles(resultfiles, filterfunc=None):
    results = []
    for resultfile in resultfiles:
        benchmarks = et.parse(resultfile).getroot()

        for benchmark in benchmarks:
            name = benchmark.attrib['name']
            n = benchmark.attrib['n']
            cold_count = benchmark.attrib.get('cold_iterations')
            peak_alloc = int(benchmark.attrib.get('peak_alloc'))
            maxrss = int(benchmark.attrib.get('maxrss'))
            ppm = benchmark.attrib.get('ppm')
            buildtype = benchmark.attrib.get('buildtype')
            configname = benchmark.attrib.get('configname')
            count = len(benchmark)

            time_per_key = defaultdict(lambda: [])
            for b in benchmark:
                for k, v in b.attrib.items():
                    time_per_key[k] .append(parse_time(v))

            stat_per_key = {}
            for k, data in time_per_key.items():
                stat_per_key[k] = statistic(data)

            item = BenchResult(name=name, ppm=ppm, buildtype=buildtype, count=count, durations=stat_per_key,
                               cold_count=cold_count, peak_alloc=peak_alloc, configname=configname, maxrss=maxrss)
            if filterfunc and not filterfunc(item):
                continue
            results.append(item)
    return results



def results_compare(results, compare_by, group_by=None, compare_val=None, show_groups=None,always_columns=["program"]):
    groups = GroupedBenches(data=results,group_by=group_by,compare_by=[compare_by])

    # Categorical groupings
    if group_by is None:
        group_by = []
    group_by = [g for g in group_by if g != compare_by]

    grouped_results, all_cmpvals, div_groups = grouping(results, compare_by=compare_by, group_by=group_by)

    common_columns= show_groups or div_groups
    compare_columns = compare_val
    more_columns = []
    for c in always_columns:
        if c not in common_columns and c not in compare_columns:
         more_columns.append(c)
    common_columns = more_columns + common_columns

    print_comparison(groups_of_results=grouped_results,
                    list_of_resultnames=all_cmpvals,
                     common_columns=common_columns,
                     compare_columns=compare_columns)





def compareby(results: Iterable[BenchResult], compare_by: str):
    results_by_group = defaultdict(lambda: [])
    for result in results:
        cmpval = get_column_data(result, compare_by)
        results_by_group[cmpval].append(result)
    return results_by_group



# TODO: dataclass?
class BenchResultGroup:
    def __init__(self, results):
        self.name = name_or_list(unique(r.name for r in results))
        self.ppm = name_or_list(unique(r.ppm for r in results))
        self.buildtype = name_or_list(unique(r.buildtype for r in results))
        self.configname = name_or_list(unique(r.configname for r in results))

        # Combine all durations to a single statistic; TODO: Should we do something like mean-of-means?
        measures = unique(k for r in results for k in r.durations.keys())
        self.durations = {m: statistic(v for r in results if m in r.durations for v in r.durations[m]._samples ) for m in measures}



class GroupedBenches:
    def __init__(self,data,group_by=None,compare_by=None):
        all_compare_keys = OrderedSet(first_defined(compare_by,['program']))
        if group_by is None:
            # Use all non-compare keys for grouping
            group_by = BenchResult.categorical_cols # ["program", "ppm", "buildtype", "configname"]
        all_group_keys = OrderedSet(group_by) .difference(all_compare_keys)

        all_compare_tuples = OrderedSet( tuple(get_column_data(result, col) for col in all_compare_keys) for result in data)
        all_group_tuples = OrderedSet( tuple(get_column_data(result, col) for col in all_group_keys) for result in data)

        # Create the matrix
        group_lists = [[[] for c in all_compare_tuples] for t in all_group_tuples]
        for d in data:
            group_key = tuple(get_column_data(d, col) for col in all_group_keys)
            group_idx = all_group_tuples.index(group_key)
            compare_key = tuple(get_column_data(d, col) for col in all_compare_keys)
            compare_idx = all_compare_tuples.index(compare_key)
            group_lists[group_idx][compare_idx].append(d)

        benchgroups = [[BenchResultGroup(c) for c in g ] for g in group_lists]

        self.compare_by=all_compare_keys
        self.compare_tuples = all_compare_tuples
        self.group_by = all_group_keys
        self.group_tuples = all_group_tuples
        self.benchgroups  = benchgroups

    def divergent_group_keys(self):
        return  divergent_keys(self.group_by,self.group_tuples)

    def divergent_compare_keys(self):
        return  divergent_keys(self.compare_by,self.compare_tuples)










# Deprecated by GroupedBenches
def grouping(results: Iterable[BenchResult], compare_by: str, group_by=None):
    """
Group benchmarks by propery

Parameters
----------
results
    List of results
compare_by
    Second order grouping categories
group_by
    First order grouping categories

Returns
-------
(grouped_results,all_cmpvals,show_groups)

grouped_results
all_cmpvals
show_groups
"""

    # TODO: allow compare_by multiple columns
    # TODO: allow each benchmark to be its own group; find description for each such "group"
    results_by_group = defaultdict(lambda: defaultdict(lambda: []))
    all_cmpvals = OrderedSet()
    for result in results:
        group = tuple(get_column_data(result, col) for col in group_by)
        cmpval = get_column_data(result, compare_by)
        all_cmpvals.add(cmpval)
        results_by_group[group][cmpval].append(result)

    grouped_results = []
    all_groups = []
    for group, group_results in results_by_group.items():
        group_cmp_results = []
        for cmpval in all_cmpvals:
            myresults = group_results.get(cmpval)
            if myresults:
                group_cmp_results.append(BenchResultGroup(myresults))
            else:
                # No values
                group_cmp_results.append(None)
            #is_unique_groups = tuple(same_or_none( g[i] for g in groups) is not None for  i in range(len(group_by)))
        grouped_results.append(group_cmp_results)
        all_groups.append(group)

    # Find all fields that could be grouped by and have different values
    show_groups = divergent_fields(group_by, results)

    return grouped_results, list(all_cmpvals), show_groups



def divergent_keys(keys,tuples):
    div_keys = []
    for i,k in enumerate(keys):
        common_value = None
        has_different_values = False
        for t in tuples:
            v = t[i]
            if common_value is None:
                common_value = v
            elif common_value == v:
                continue
            else:
                has_different_values = True
                break
        if has_different_values:
            div_keys.append(k)
    return div_keys


def divergent_fields(group_by, results):
    show_groups = []
    for col in group_by:
        common_value = None
        has_different_values = False
        for result in results:
            val = get_column_data(result, col)
            if common_value is None:
                common_value = val
            elif common_value == val:
                continue
            else:
                has_different_values = True
                break
        if has_different_values:
            show_groups.append(col)
    return show_groups



def evaluate(resultfiles):
    results = load_resultfiles(resultfiles)

    stats_per_key = defaultdict(lambda: [])
    for r in results:
        for k, stat in r.durations.items():
            stats_per_key[k] .append(stat)

    summary_per_key = {}  # mean of means
    for k, data in stats_per_key.items():
        summary_per_key[k] = statistic(d.mean for d in data)

    table = Table()

    def count_formatter(v: int):
        s = str(v)
        return StrAlign(StrColor(str(v), colorama.Fore.BLUE), printlength(s))

    def ppm_formatter(s: str):
        return getPPMDisplayStr(s)

    table.add_column('program', title=StrColor("Benchmark", colorama.Fore.BWHITE), formatter=path_formatter)
    table.add_column('ppm', title="PPM", formatter=ppm_formatter)
    table.add_column('buildtype', title="Buildtype")
    table.add_column('n', title=StrColor("Repeats", colorama.Style.BRIGHT), formatter=count_formatter)
    for k, summary in summary_per_key.items():
        table.add_column(k, title=StrColor(getMeasureDisplayStr(k), colorama.Style.BRIGHT),
                         formatter=duration_formatter(summary.minimum, summary.maximum))

    for r in results:
        # TODO: acceltime doesn't always apply
        table.add_row(program=r.name, ppm=r.ppm, buildtype=r.buildtype, n=r.count, **r.durations)

    table.print()






def getMeasureDisplayStr(s: str):
    return {'walltime': "Wall", 'usertime': "User", 'kerneltime': "Kernel",
            'acceltime': "CUDA Event",
            'cupti': "nvprof", 'cupti_compute': "nvprof Kernel", 'cupti_todev': "nvprof H->D", 'cupti_fromdev': "nvprof D->H"}.get(s, s)


def getPPMDisplayStr(s: str):
    return {'serial': "Serial", 'cuda': "CUDA", 'omp_parallel': "OpenMP parallel",
            'omp_task': "OpenMP task", 'omp_target': "OpenMP Target Offloading"}.get(s, s)



def print_comparison(groups_of_results, list_of_resultnames, common_columns=["program"], compare_columns=[]):
    """
Print a benchmark result table.

Parameters
----------
results_of_groups
    Matrix of BenchResults or BenchResultGroups. Each major represents a row in the output table. Minors of the same major represent the results to be compared to each other. Benchmarks in a BenchResultGroup are to be summarized.
list_of_resultnames
    ?
common_columns
    Columns where the results of all minors of the same row are to be summerized into a single columns.
compare_columns
    Columns where the results of the minors of the same row are to be compared; each minor gets its own subcolumn.
"""

    table = Table()

    for col in common_columns:
        if col == "program":
            table.add_column(col, title=StrAlign(StrColor("Benchmark", colorama.Fore.BWHITE),
                             pos=StrAlign.CENTER), formatter=program_formatter)
        else:  # TODO: proper column name
            table.add_column(col, title=StrAlign(StrColor(col, colorama.Fore.BWHITE), pos=StrAlign.CENTER))

    for j, col in enumerate(compare_columns):
        supercolumns = []
        table.add_column(col, StrAlign(StrColor(getMeasureDisplayStr(col),
                         colorama.Style.BRIGHT), pos=StrAlign.CENTER))
        for i, resultname in enumerate(list_of_resultnames):  # Common title
            sol = f"{col}_{i}"
            supercolumns.append(sol)
            table.add_column(sol, title=StrAlign(StrColor(resultname, colorama.Style.BRIGHT),
                             pos=StrAlign.CENTER), formatter=duration_formatter())
        table.make_supercolumn(f"{col}", supercolumns)

    for result in groups_of_results:
        representative = result[0]  # TODO: collect all occuring group values
        data = dict()
        for col in common_columns:
            data[col] = get_column_data(representative, col)
        for col in compare_columns:
            for i, resultname in enumerate(list_of_resultnames):
                data[f"{col}_{i}"] = get_column_data(result[i], col)
        table.add_row(**data)

    table.print()




class HtmlWriter:
    def __init__(self,fd):
        self.indent = 0
        self.fd = fd
        self.nest = []

    def print(self,*args):
        print(' ' * (2* self.indent),end='', file=self.fd)
        print(*args,file=self.fd)

    def escaped(self,*args,quote=False):
        print(' ' * (2* self.indent),end='', file=self.fd)
        esc = (html.escape(str(arg),quote=quote) for arg in args)
        print(*esc,file=self.fd)


    def enter(self,tag,**props):
        proplist = (f' {k}="{html.escape(v,quote=True)}"' for k,v in props.items())
        self.print(f"<{tag}{''.join(proplist)}>")
        self.nest.append(tag)
        self.indent += 1

    def leave(self):
        tag = self.nest.pop()
        self.indent -= 1
        self.print(f"</{tag}>")

    def tag(self,tag):
        class TagContextmanager:
            def __init__(self,html,tag):
                    self.html=html
                    self.tag=tag
            def __enter__(self):
                self.html.enter(self.tag)
            def __exit__(self, exc_type, exc_value, exc_traceback):
                self.html.leave()
            def print(self,*args):
                self.html.print(*args)
            def tag(self,tag,**props):
                return self.html.tag(tag,**props)
        return TagContextmanager(self,tag)



def save_report(results,filename):
    filename = mkpath(filename)

    with filename.open("w+") as f:
        html = HtmlWriter(f)
        make_report(html,results)





def results_speedupplot(results, baseline, group_by=None, compare_by=None,value_key='walltime'):
    groups =  GroupedBenches(data=results,group_by=group_by,compare_by=compare_by)
    div_group_keys = groups.divergent_group_keys()
    div_compare_keys = groups.divergent_compare_keys()

    def make_group_label(t: tuple): # TODO: at least one label
        return ', '.join(v for i, v in enumerate(t) if groups.group_by[i] in div_group_keys )

    def make_compare_label(t: tuple):
        return ', '.join(v for i, v in enumerate(t) if groups.compare_by[i] in div_compare_keys )

    labels = [make_group_label(g) for g in groups.group_tuples]
    assert baseline in groups.compare_tuples
    baseline_compare_idx = groups.compare_tuples.index(baseline)

    left = 1
    right = 0.5
    numgroups = len(groups.group_tuples)
    benchs_per_group = len(groups.compare_tuples) -1
    barwidth = 0.3
    groupwidth = 0.2 + benchs_per_group * barwidth
    width = left + right + groupwidth * numgroups
    fig, ax = plt.subplots(figsize=(width, 10))
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [c['color'] for j, c in zip(range(benchs_per_group), prop_cycle)]  # TODO: Consider seaborn palettes
    compare_tuples_without_baseline = [c for c in groups.compare_tuples if c != baseline]

    fig.subplots_adjust(left=left / width, right=1 - right / width, top=0.95, bottom=0.25)

    for group_idx, group_key in enumerate(groups.group_tuples):
        group_data = groups.benchgroups[group_idx]
        baseline_result = group_data[baseline_compare_idx]
        baseline_stat = get_column_data(baseline_result,value_key) # TODO: Skip group if baseline is missing
        group_data_without_baseline = [b for i,b in enumerate(group_data) if i!=baseline_compare_idx]
        baseline_mean =baseline_stat.mean
        nonempty_results = [(j,r) for j,r in enumerate(group_data_without_baseline) if r]
        benchs_this_group = len(nonempty_results)


        for i,( j, benchstat )in enumerate(nonempty_results):
            stat = get_column_data(benchstat,value_key)

            # Skip bar if there is no statistic
            if stat:
                mean = stat.mean
                speedup = baseline_mean / mean

                rel = (i - benchs_this_group / 2.0 + 0.5) * barwidth
                abserr = stat.abserr()
                kwargs = {}
                if abserr is not None:
                    kwargs['yerr'] = abserr/baseline_mean
                bar = ax.bar(x=group_idx * groupwidth + rel, height=speedup,width=barwidth,color = colors[j],bottom=1,**kwargs)


    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    for j, (c, label) in enumerate(zip(colors, compare_tuples_without_baseline)):
        # Dummy item to add a legend handle; like seaborn does
        rect = plt.Rectangle([0, 0], 0, 0,
                             # linewidth=self.linewidth / 2,
                             # edgecolor=self.gray,
                             facecolor=c,
                             label=label)
        ax.add_patch(rect)

    # TODO: Compute conf_intervals consistently like the table, preferable using the student-t test.
    # x.grid(linestyle='--',axis='y')
    ax.set(
        axisbelow=True,  # Hide the grid behind plot objects
        xlabel='Benchmark',
        ylabel='Walltime [s]',
    )
    ax.set_yscale('log',base=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xticks([groupwidth * i for i in range(len(labels))])
    ax.set_xticklabels(labels, rotation=20, ha="right", rotation_mode="anchor")

    plt.legend()

    fig =  plt.gcf()
    i = io. StringIO()
    plt.savefig(i, format="svg")
    fig.canvas.draw_idle()


    i.seek(0)
    s = et.canonicalize(from_file=i) # Remove <?xml> boilerplate
    return s


def make_report(html,results):
    html.print("<!DOCTYPE html>")
    with html.tag("html"):
        with html.tag("head"):
            html.print("<title>Benchmark Report</title>")
        with html.tag("body"):
            html.print("<h1>Benchmark Report</h1>")

            html.print("<h2>Speedup Relative to serial</h2>")
            figdata = results_speedupplot(results,baseline=('serial',),compare_by=['ppm'])
            html.print(figdata)

            html.print("<h2>All Results</h2>")
            with html.tag("table"):
                with html.tag("tr"):
                    html.print("<td>Program</td>")
                    html.print("<td>PPM</td>")
                    html.print("<td>Buildtype</td>")
                    html.print("<td>Config</td>")
                    html.print("<td>Walltime</td>")

                for r in results:
                    with html.tag("tr"):
                        with html.tag("td"):
                            html.escaped(r.name)
                        with html.tag("td"):
                            html.escaped(r.ppm)
                        with html.tag("td"):
                            html.escaped(r.buildtype)
                        with html.tag("td"):
                            html.escaped(r.configname)
                        with html.tag("td"):
                            html.escaped(r.durations['walltime'].mean)


