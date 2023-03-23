# -*- coding: utf-8 -*-


"""
Evaluate, compare, and report for a set of resultfile 
"""

import io
from collections import defaultdict
import colorama
import xml.etree.ElementTree as et
from typing import Iterable
import html
import pathlib
from itertools import count
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from cycler import cycler
from .util.cmdtool import *
from .util.orderedset import OrderedSet
from .util.support import *
from .table import *
from .stat import *
from .common import *








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


import dateutil


#TODO: dataclass
class BenchResult:
    categorical_cols = ['program', 'ppm', 'buildtype', 'configname', 'timestamp']
    numerical_cols = ['walltime']
    def __init__(self, name: str, ppm: str, buildtype: str, configname: str, timestamp: str,
                 count: int, durations, maxrss=None, cold_count=None, peak_alloc=None):
        # self.bench=bench
        self.name = name
        self.ppm = ppm
        self.buildtype = buildtype 
        self.configname = configname
        self.timestamp = dateutil. parser.parse( timestamp) if isinstance(timestamp,str) else timestamp
        self.count = count
        # self.wtime=wtime
        # self.utime=utime
        # self.ktime=ktime
        # self.acceltime=acceltime
        self.durations = durations
        self.maxrss = maxrss
        self.cold_count = cold_count
        self.peak_alloc = peak_alloc


# TODO: dataclass?
class BenchResultSummary:
    """A summery of multiple BenchResults with the same API as BenchResult"""
    def __init__(self, results): # TODO: Detect if any lsit element is itself a BenchResultSummary and expand it
        self.name = name_or_list(unique(r.name for r in results))
        self.ppm = name_or_list(unique(r.ppm for r in results))
        self.buildtype = name_or_list(unique(r.buildtype for r in results))
        self.configname = name_or_list(unique(r.configname for r in results))
        self.timestamp  = name_or_list(unique(r.timestamp for r in results))

        # Combine all durations to a single statistic; TODO: Should we do something like mean-of-means?
        measures = unique(k for r in results for k in r.durations.keys())
        self.durations = {m: statistic(v for r in results if m in r.durations for v in r.durations[m]._samples ) for m in measures}




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
    if colname == "timestamp":
        return result.timestamp
    assert False, "TODO: Add to switch of use getattr"

def get_summary_data(result: BenchResultSummary, colname: str):
    return get_column_data(result, colname)


def getColumnFormatter(colname: str):
    if colname == 'program':
        return program_formatter
    elif colname in BenchResult.numerical_cols:
        return duration_formatter() # TOOD: Get best/worst
    return None




def formatColumnVal(colname:str, val):
    formatter = getColumnFormatter(colname)
    if formatter:
        return formatter(val)
    return         str(val)


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




def getHTMLFromatter(col: str):
    def str_html_formatter(v):
        return html.escape(str(v))
    def duration_formatter(stat):
        assert isinstance(stat,Statistic)
        v = stat.mean
        if v >= 1:
            return f'{v:.2f}'
        if v * 1000 >= 1:
            return f'{v*1000:.2f}<span class="timeunit">ms</span>'
        if v * 1000 * 1000 >= 1:
            return f'{v*1000*1000:.2f}<span class="timeunit">µs</span>'
        if v * 1000 * 1000 * 1000 >= 1:
            return f'{v*1000*1000*1000:.2f}<span class="timeunit">ns</span>'

    if col in BenchResult.numerical_cols:
        return duration_formatter
    return str_html_formatter




def getPlaintextFormatter(col: str):
    def timestamp_plaintext_formatter(v):
        assert isinstance(v,datetime.datetime)
        return f'{v.astimezone():%c}'
    def str_plaintext_formatter(v):
        return str(v)
    def duration_plaintext_formatter(stat):
        assert isinstance(stat,Statistic)
        v = stat.mean
        if v >= 1:
            return f'{v:.2f}'
        if v * 1000 >= 1:
            return f'{v*1000:.2f} ms'
        if v * 1000 * 1000 >= 1:
            return f'{v*1000*1000:.2f} µs'
        if v * 1000 * 1000 * 1000 >= 1:
            return f'{v*1000*1000*1000:.2f} ns'

    if col == 'timestamp':
        return timestamp_plaintext_formatter
    if col in BenchResult.numerical_cols:
        return duration_plaintext_formatter
    return  str_plaintext_formatter















def load_resultfiles(resultfiles, filterfunc=None):
    results = []
    for resultfile in resultfiles:
        benchmarks = et.parse(resultfile).getroot()

        for benchmark in benchmarks:
            name = benchmark.attrib['name']
            n = benchmark.attrib['n']
            cold_count = benchmark.attrib.get('cold_iterations')
            peak_alloc = benchmark.attrib.get('peak_alloc')
            if peak_alloc:
                peak_alloc = int(peak_alloc)
            maxrss = benchmark.attrib.get('maxrss')
            if maxrss:
                maxrss = int(maxrss)
            ppm = benchmark.attrib.get('ppm')
            buildtype = benchmark.attrib.get('buildtype')
            configname = benchmark.attrib.get('configname')
            timestamp = benchmark.attrib.get('timestamp')
            count = len(benchmark)

            time_per_key = defaultdict(lambda: [])
            for b in benchmark:
                for k, v in b.attrib.items():
                    time_per_key[k] .append(parse_time(v))

            stat_per_key = {}
            for k, data in time_per_key.items():
                stat_per_key[k] = statistic(data)

            item = BenchResult(name=name, ppm=ppm, buildtype=buildtype, timestamp=timestamp, count=count, durations=stat_per_key,
                               cold_count=cold_count, peak_alloc=peak_alloc, configname=configname, maxrss=maxrss)
            if filterfunc and not filterfunc(item):
                continue
            results.append(item)
    return results



def default_columns(groups,compare_by,always_columns,never_columns):
        columns = OrderedSet()
        columns.union_update(always_columns)  # Try to put these to the front
        columns.union_update(groups.divergent_categories)
        columns.union_update( groups.nonempty_vals() )
        columns.difference_update(OrderedSet(compare_by).difference(always_columns) )
        columns.difference_update( never_columns)
        columns= list(columns)
        def enforce_order(x):
            if x == 'program':
                return 0
            if x in BenchResult.categorical_cols:
                return 1
            return 2
        columns.sort(key=enforce_order)
        return columns



def results_compare(results, compare_by=None, group_by=None, compare_val=None, show_groups=None, always_columns=['program'],never_columns=[],columns=None):
    groups = GroupedBenches(data=results,group_by=group_by,compare_by=compare_by)
    compare_by = compare_by or []

    if columns is None:
        columns = default_columns(groups,compare_by=compare_by, always_columns=always_columns,never_columns=never_columns)

    compare_columns=set()
    if compare_by:
        compare_columns = set(columns) & (set(groups.divergent_compare_keys()) | set(BenchResult.numerical_cols))

    print_comparison(groups, columns= columns,compare_columns=compare_columns )

    # Categorical groupings
    #if group_by is not None:
    #    group_by = [g for g in group_by if g != compare_by] 

    #grouped_results, all_cmpvals, div_groups = grouping(results, compare_by=compare_by, group_by=group_by)

    #common_columns= show_groups or div_groups
    #compare_columns = compare_val
    #more_columns = []
    #for c in always_columns:
    #    if c not in common_columns and c not in compare_columns:
    #     more_columns.append(c)
    #common_columns = more_columns + common_columns
    #
    #print_comparison(groups_of_results=grouped_results,
    #                 list_of_resultnames=all_cmpvals,
    #                 common_columns=common_columns,
    #                 compare_columns=compare_columns)





def compareby(results: Iterable[BenchResult], compare_by: str):
    results_by_group = defaultdict(lambda: [])
    for result in results:
        cmpval = get_column_data(result, compare_by)
        results_by_group[cmpval].append(result)
    return results_by_group



def colsortkey(item, col):
            if col == 'program':
                # Return programs alphabetically
                return item
            if col == 'ppm':
                return {  'serial': 0, 'omp_parallel': 1, 'omp_task': 2, 'omp_target': 3, 'cuda': 4 }.get(item, 5)
            return 0 # Keep the original order of everything else


def sort_keys(key_tuples, compare_by):
    def keyfunc(x):
        return tuple( colsortkey(e, col) for col, e in zip(compare_by,x)  )
    return sorted (key_tuples, key=keyfunc)


def sort_results(benchresults):
    # TODO: Should be configurable
    def keyfunc(x):
        return ( colsortkey(x.name, 'program') , colsortkey(x.configname, 'configname') ,colsortkey(x.configname, 'buildtype') , colsortkey(x.ppm, 'ppm')  )
    return sorted (benchresults, key=keyfunc)


class GroupedBenches:
    def __init__(self,data,group_by=None,compare_by=None): #TODO: Add another level to make comparable items stay close in the table (and add best/worst color per bucket); How to present in boxplot? 3rd dimension?
        """Group results into two levels of buckets. 

group_by buckets are meant to each have its own table for, or if a box blot, having different positions on the x-axis.

compare_by is meant to be shown on the same table row but each bucket having a separate column for its summerized data, or if a box boxplot, having the bars directly next to each other.

If group_by is specified, but compare_by is not, then each group_by bucket has only a single compare_by bucket where the entrire group_by bucket is summerized (equivalent to compare_by=[])

If group_by is not specified, but compare_by is, then group by all other (non-compared) benchmark categories.

If both are unspecified, every result gets its own group_by backet with a single compare_by bucket. 
        """

        # Sort data
        data = sort_results(data)

        # Special case: No grouping as all, no need to summerize
        if group_by is None and compare_by is None:
            # Each benchmark has its own group with just a single comparison category
            self.compare_by=[]
            self.compare_tuples = [()]
            self.group_by = None
            self.group_tuples =  [(i,) for i,d  in enumerate(data)]
            self.benchgroups = [[d] for d in data]
            self.groupsummary = [d for d in data]
            return 


        all_compare_keys = OrderedSet(first_defined(compare_by,[])) 
        all_compare_tuples = OrderedSet(tuple(get_column_data(result, col) for col in all_compare_keys) for result in data)
        all_compare_tuples = sort_keys(all_compare_tuples,compare_by=all_compare_keys)

        if group_by is None:
              all_group_keys = OrderedSet(BenchResult.categorical_cols )  .difference(all_compare_keys)
        else:
            all_group_keys= group_by          
        all_group_tuples = OrderedSet(tuple(get_column_data(result, col) for col in all_group_keys) for result in data)

   

        # Create the matrix
        # TODO: Consider itertools.groupby (twice)
        group_lists = [[[] for c in all_compare_tuples] for t in all_group_tuples]
        for d in data:
            group_key = tuple(get_column_data(d, col) for col in all_group_keys)
            group_idx = all_group_tuples.index(group_key)
            compare_key = tuple(get_column_data(d, col) for col in all_compare_keys)
            compare_idx = all_compare_tuples.index(compare_key)
            group_lists[group_idx][compare_idx].append(d)

        benchgroups = [[BenchResultSummary(c) for c in g] for g in group_lists]
        groupsummary = [BenchResultSummary([b for c in g for b in c]) for g in group_lists]

        self.compare_by=all_compare_keys
        self.compare_tuples = all_compare_tuples
        self.group_by = all_group_keys
        self.group_tuples = all_group_tuples
        self.benchgroups  = benchgroups
        self.groupsummary =groupsummary

    @property
    def all(self):
        for g in self.benchgroups:
            for c in g:
                yield c

    def divergent_group_keys(self):
        return  divergent_keys(self.group_by ,self.group_tuples)
    
    def divergent_compare_keys(self):
        return  divergent_keys(self.compare_by,self.compare_tuples)

    @property
    def divergent_categories(self):
        divcat = []
        for cat in BenchResult.categorical_cols:
            common_val = None
            for g in self.benchgroups:
                summary = BenchResultSummary( g)
                val = get_column_data(summary, cat)
                if val is None:
                    continue 
                if common_val is None:
                    common_val = val
                elif val != common_val:
                    break
            else:
                continue
            divcat .append(cat)
        return divcat

    def nonempty_vals(self):
        nonemptykeys = []
        for cat in BenchResult.numerical_cols:
            for benchresult in (b for g in self.benchgroups for b in g):
                   val = get_column_data(benchresult, cat) 
                   if val :
                       nonemptykeys.append(cat)
                       break
        return nonemptykeys











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
    if group_by is  None:
        return [] 

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






# TODO: Rename: getColumnDisplayString
def getMeasureDisplayStr(s: str):
    return {'program': "Benchmark",
            'ppm': "PPM",
            'buildtype':  "Buildtype",
            'configname': "Configuration",
            'walltime': "Wall", 
            'usertime': "User", 
            'kerneltime': "Kernel",
            'acceltime': "CUDA Event",
            'cupti': "nvprof", 
            'cupti_compute': "nvprof Kernel", 
            'cupti_todev': "nvprof H->D", 
            'cupti_fromdev': "nvprof D->H"}.get(s, s)


def getPPMDisplayStr(s: str):
    return {'serial': "Serial", 'cuda': "CUDA", 'omp_parallel': "OpenMP parallel",
            'omp_task': "OpenMP task", 'omp_target': "OpenMP Target Offloading"}.get(s, s)



def print_comparison(benchgroups:GroupedBenches,columns,compare_columns):
    """Print a benchmark result table."""
    table = Table()

    # Define the table layout
    for col in columns:
        if col in compare_columns:
                # Make a supercolumn for value comparisons
                subcolumns = []
                table.add_column(col, StrAlign(StrColor(getMeasureDisplayStr(col), colorama.Style.BRIGHT), pos=StrAlign.CENTER))
                for  i, resulttuple in enumerate( benchgroups.compare_tuples): 
                    sol = f"{col}_{i}"
                    subcolumns.append(sol)
                    resultname = ','.join( formatColumnVal(ccat, resulttuple[i]) for i,ccat in enumerate(benchgroups.compare_by) )
                    table.add_column(sol, title=StrAlign(StrColor(resultname, colorama.Style.BRIGHT),  pos=StrAlign.CENTER), formatter=getColumnFormatter(col))
                table.make_supercolumn(f"{col}", subcolumns)
        else:
                table.add_column(col, title=StrAlign(StrColor(getMeasureDisplayStr(col), colorama.Fore.BWHITE),  pos=StrAlign.CENTER), formatter=getColumnFormatter(col))


    # Set the table data
    for rowsummery,row in zip(benchgroups.groupsummary, benchgroups.benchgroups):
        data = dict()
      
        for col in columns:
            if col in compare_columns:
                for i, resulttuple in enumerate(row): 
                    val = get_summary_data(row[i], col)
                    data[f"{col}_{i}"] = val
            else:
                data[col] =  get_summary_data(rowsummery, col)
        table.add_row(**data)

    table.print()









def results_speedupplot(groups:GroupedBenches, data_col, logscale=True,baseline_cmpval=None,relcompare=True):
    """Create a results plot

:param groups: The data, grouped

:param data_col: The data to use for the y-axis

:param logscale: If true, use a logarithmic y-axis

:param baseline_cmpval: If set, the compare_tuple the others are compared to; 
                        If not set, show absolute values

:param relcompare: If True, show the ratio between the value and the baseline 
                   If False, show the different to the baseline value
                   Only meaningful when baseline_cmpval is set
"""

    if groups.group_by:
        label_groups =  groups.group_by
    else:
        label_groups = BenchResult.categorical_cols

    # Find the diverging categories
    label_groups = divergent_keys(label_groups, [ tuple( get_summary_data(r, col) for col in label_groups ) for r in  groups. groupsummary])
    if not label_groups:
        label_groups = ['program']

    def make_group_label(s,g):
            return ', '.join( getPlaintextFormatter( col)( get_summary_data(s, col)) for col in label_groups )

    labels = [make_group_label(s,g) for s,g in zip (groups. groupsummary, groups.benchgroups)]


    
    leftmargin = 1
    rightmargin = 0.5
    numgroups = len(groups.group_tuples)
    if baseline_cmpval:
        # Hide the baseline bar itself, it would but just zero (or one on logscale) anyway
        compare_tuples_without_baseline = [c for c in groups.compare_tuples if  c  != baseline_cmpval]
    else:
        compare_tuples_without_baseline = groups.compare_tuples
    benchs_per_group = len(compare_tuples_without_baseline)
    barwidth = 0.4
    groupwidth = 0.3 + benchs_per_group * barwidth
    plotwidth = groupwidth * numgroups
    width = leftmargin + rightmargin + plotwidth
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [c['color'] for j, c in zip(range(benchs_per_group), prop_cycle)]





    fig, ax = plt.subplots(figsize=(width, 10))
    fig.subplots_adjust(left=leftmargin / width, right=1 - rightmargin / width, top=0.95, bottom=0.25)


    for group_idx, (group_summary, group_data) in  enumerate(zip(groups.groupsummary, groups.benchgroups)):
        baseline_idx = None
        if baseline_cmpval:
            # FIXME: What if missing?
            [(baseline_idx, baseline_result)] = ((i,c) for i,(t,c) in enumerate(zip(groups.compare_tuples , group_data)) if t == baseline_cmpval)
            baseline_stat = get_column_data(baseline_result,data_col) 
            if not baseline_stat:
                log.warn(f"No baseline {baseline_cmpval}; skipping group f{group_idx}")
                continue
            #group_data_without_baseline = [b for i,b in enumerate(group_data) if i!=baseline_compare_idx]
            baseline_mean = baseline_stat.mean
             
        #nonempty_results = [(j,r) for j,r in enumerate(group_data_without_baseline) if r]
        #benchs_this_group = len(nonempty_results)

        for compare_idx, compare_data in enumerate(d for i,d in enumerate( group_data) if i != baseline_idx ):
            stat = get_column_data(compare_data,data_col)
            if not stat:
                # No bar for missing measurement
                continue

            if baseline_cmpval:
                if relcompare:
                    val = baseline_mean / stat.mean
                else:
                    val = stat.mean - baseline_mean
            else:
                val = stat.mean

            # TODO: Error bars
            #    abserr = stat.abserr()
            #    kwargs = {}
            #    if abserr is not None:
            #        kwargs['yerr'] = abserr/baseline_mean


            rel = (compare_idx - benchs_per_group / 2.0 + 0.5) * barwidth
            xpos = group_idx * groupwidth + rel
            if baseline_cmpval:
                bar = ax.bar(x=xpos, height=val,width=barwidth,color = colors[compare_idx],bottom=1)
            else:
                bar = ax.boxplot( stat.samples  , positions=[xpos],
                             notch=True, showmeans=False, showfliers=True, sym='+',
                             widths=barwidth,
                             patch_artist=True,  # fill with color
                             )

    if logscale:
        ax.set_yscale('log')
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    # Set bar colors
    for c, label in zip(colors, compare_tuples_without_baseline):
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

    ax.legend()
    return fig


class ReportSection:
    pass


class AllResultsSection(ReportSection):
    name = 'all-results'
    title = "All Results (Table)"
    
    def __init__(self, groups,columns=None,compare_columns=[]):
        super().__init__()

        if columns is None:
            columns = default_columns(groups,compare_by=[], always_columns=['program'],never_columns=[])


        self.groups = groups
        self.columns = columns



    @cached_generator
    def content(self):
        benchgroups = self.groups
        columns = self.columns
        compare_columns =[]

        yield '<table class="table">'

        # Print the table header
        yield "<thead><tr>"
        for col in columns:
            if col in compare_columns:
                    yield f'<td colspan="{len(columns)}">{getMeasureDisplayStr(col)}</td>'
            else:
                    yield f"<td>{getMeasureDisplayStr(col)}</td>"
        if compare_columns:
            yield "</tr><tr>"
            for col in columns:
                if col in compare_columns:
                    for  i, resulttuple in enumerate( benchgroups.compare_tuples): 
                        resultname = ','.join( formatColumnVal(ccat, resulttuple[i]) for i,ccat in enumerate(benchgroups.compare_by) )
                        yield f"<td>{resultname}</td>"
                else:
                    yield f"<td></td>"
        yield "</tr></thead>"

        # Emit table data
        
        for rowsummery,row in zip(benchgroups.groupsummary, benchgroups.benchgroups):
            yield "<tr>"
            for col in columns:
                if col in compare_columns:
                    for i, resulttuple in enumerate(row): 
                        yield f'<td>{formatColumnVal(col, get_summary_data(row[i], col))}</td>'
                else:
                    yield f'<td>{getHTMLFromatter(col)( get_summary_data(rowsummery, col))}</td>'
            yield "</tr>"
        yield "</table>"



class WalltimePlotSection(ReportSection):
    name = 'walltime-plot'
    title = "Walltime Plot"

    
    def __init__(self, groups,columns=None,compare_columns=[]):
        super().__init__()

        if columns is None:
            columns = default_columns(groups,compare_by=[], always_columns=['program'],never_columns=[])

        self.groups = groups
        self.columns = columns




    @cached_generator
    def content(self):
        benchgroups = self.groups
        columns = self.columns
        compare_columns = []

        fig = results_speedupplot(benchgroups, data_col='walltime',logscale=False)
        s = fig_to_svg(fig)
        yield s
 


def fig_to_svg(fig):
        i = io. StringIO()
        #plt.savefig(i, format="svg")
        fig.savefig(i, format='svg')
        fig.canvas.draw_idle()
        i.seek(0)
        s = et.canonicalize(from_file=i) # Remove <?xml> boilerplate
        return s


class SpeedupPlotSection(ReportSection):
    name = 'speedup-plot'
    title = "Speedup Plot"

    
    def __init__(self, groups,compare_col,base_cat):
        assert groups.compare_by == [compare_col]

        super().__init__()
        self.groups = groups
        self.compare_col=compare_col
        self.base_cat = base_cat



    @cached_generator
    def content(self):
        benchgroups = self.groups

        fig = results_speedupplot(benchgroups, data_col='walltime', baseline_cmpval=(self.base_cat,),relcompare=True,logscale=True)
        s = fig_to_svg(fig)
        yield s
 



def make_report(results):
    groups = GroupedBenches(data=results)
    resultssec = AllResultsSection(groups)
    resultsplotsec = WalltimePlotSection(groups)
    sections =  [resultssec, resultsplotsec]

    ppms = unique( c.ppm for c in  results  )
    if len(ppms)>= 2:
        walltimecompare  = GroupedBenches(data=results,compare_by=['ppm'])
        for base in ppms:
            speedupplotsec = SpeedupPlotSection(walltimecompare,compare_col='ppm',base_cat=base)
            sections.append(speedupplotsec)
    
    yield """
<!DOCTYPE html>
<html>
    <head>
        <title>Benchmark Report</title>
	    <style>
		body {
			font-family: sans-serif;
			font-size: 14px;
			line-height: 1.5;
			margin: 0;
			padding: 0;
		}

        /* CSS grid layout */
		.container {
			display: flex;
			flex-wrap: wrap;
			margin: 0 auto;
			padding: 20px;
            column-gap: 15px;
            display: grid;
            grid-template-columns: 15em minmax(0,1fr);
		}

        /* CSS grid column 1: navigation */
        .toc-container {  /* Container in which the sticky div can move */
            display: block;
            height: 100%;
        }
		.toc {
			background-color: #f1f1f1;
			border-radius: 5px;
			box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);
			flex-basis: 20%;
            top: 20px;
			position: sticky;
		}
		.toc ul {
			list-style: none;
			margin: 0;
			padding: 0;
		}
		.toc li {
			margin-bottom: 10px;
		}

        /* CSS grid column 2: content */
		.table-container {
			flex-basis: 80%;
		}
		.table {
			border-collapse: collapse;
			margin-top: 20px;
			width: 100%;
		}
		.table th,
		.table td {
			border: 1px solid #ddd;
			padding: 8px;
			text-align: left;
		}
		.table th {
			background-color: #f2f2f2;
			font-weight: bold;
		}
		.table tr:nth-child(even) {
			background-color: #f9f9f9;
		}

        .timeunit {
            color: #AAAAAA;
        }
	</style>
    </head>
    <body>
      <div class="container">
      <div class="toc-container">
		<div class="toc">
			<h2>Sections</h2>
			<ul>"""
    for s in sections:
        yield f'<li><a href="#{s.name}">{s.title}</a></li>'

    yield """
			</ul>
		</div>
        </div>
        <div class="table-container">
            <h1>Benchmark Report</h1>"""

    for s in sections:
        yield f'<h2 id="{s.name}">{s.title}</h2>'
        yield from s.content
      

    yield """</div>
        </div>
    </body>
</html>
"""






def save_report(results,filename):
    filename = mkpath(filename)

    with filename.open("w+") as f:
        for  line in make_report(results):
            print(line, file=f)


