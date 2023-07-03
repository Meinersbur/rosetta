# -*- coding: utf-8 -*-

import sys
if not sys.version_info >= (3, 9):
    print("Requires python 3.9 or later", file=sys.stderr)
    print(f"Python interpreter {sys.executable} reports version {sys.version}", file=sys.stderr)
    sys.exit(1)


import typing
from collections import defaultdict
import datetime
import pathlib
import argparse
import re
import logging as log
import typing

from .builder import *
from . import registry
from .util.cmdtool import *
from .util.orderedset import OrderedSet
from .util.support import *
from .util import invoke
from .common import *
from .filtering import *


configsplitarg = re.compile(r'((?P<configname>[a-zA-Z0-9_]+)\:)?(?P<arg>.*)')
configsplitdef = re.compile(
    r'((?P<configname>[a-zA-Z0-9_]+)\:)?(?P<defname>[a-zA-Z0-9_]+)(\=(?P<defvalue>.*))?')
configsplitdefrequired = re.compile(
    r'((?P<configname>[a-zA-Z0-9_]+)\:)?(?P<defname>[a-zA-Z0-9_]+)\=(?P<defvalue>.*)')
configppm = re.compile(
    r'((?P<configname>[a-zA-Z0-9_]+)\:)?(?P<ppm>[a-zA-Z\-]+)')


def parse_build_configs(args, implicit_reference):
    def parse_arglists(l):
        raw = defaultdict(lambda: [])
        if l is None:
            return raw
        for arg in l:
            m = configsplitarg.fullmatch(arg)
            configname = m.group('configname') or ''
            value = m.group('arg')
            raw[configname].append(value)
        return raw

    def parse_deflists(l, valrequired):
        raw = defaultdict(lambda: dict())
        if l is None:
            return raw
        for arg in l:
            m = (configsplitdefrequired if valrequired else configsplitdefrequired).fullmatch(
                arg)
            configname = m.group('configname') or ''
            defname = m.group('defname')
            defvalue = m.group('defvalue')  # Can be none
            raw[configname][defname] = defvalue
        return raw

    cmake_arg = parse_arglists(args.cmake_arg)
    cmake_def = parse_deflists(args.cmake_def, valrequired=True)
    compiler_arg = parse_arglists(args.compiler_arg)
    compiler_def = parse_deflists(args.compiler_def, valrequired=False)
    ppm = parse_arglists(args.ppm)

    specified_keys = OrderedSet()
    specified_keys |= cmake_arg.keys()
    specified_keys |= cmake_def.keys()
    specified_keys |= compiler_arg.keys()
    specified_keys |= compiler_def.keys()
    specified_keys |= ppm.keys()

    selected_keys = OrderedSet()
    selected_keys |= [k for k in specified_keys if k]
    selected_keys |= (args.config or [])

    # Use single config if no "CONFIG:" is specified
    if not selected_keys:
        selected_keys.add('')
        specified_keys.add('')
    if implicit_reference:
        # TODO: Reference configuration (only reference PPM)
        selected_keys.add("REF")
        # specified_keys.add('REF')

    configs = []
    for k in selected_keys:
        # TODO: Handle duplicate defs (specific override general)
        configs.append(BuildConfig(name=k, ppm=ppm[''] + ppm[k], cmake_arg=cmake_arg[''] + cmake_arg[k], cmake_def=cmake_def[''] | cmake_def[k],
                       compiler_arg=compiler_arg[''] + compiler_arg[k], compiler_def=compiler_def[''] | compiler_def[k], usecur=not k in specified_keys))

    return configs


def print_verbose(*args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def invoke_verbose(*args, **kwargs):
    if verbose:
        invoke.diag(*args, **kwargs)
    else:
        invoke.run(*args, **kwargs)


def subcommand_default_actions(args):
    # Determine actions
    any_explicit_action = args.clean or args.configure or args.build or args.probe or args.verify or args.bench or args.evaluate
    if not any_explicit_action:
        # No explicit action selected: Use default (configure -> build -> bench -> evaluate)
        args.bench = first_defined(args.bench, True)
    # else:
    #    # Don't execute by default when some other action was explicitly defined
    #    args.bench = first_defined(args.bench , False)

    # If at least one primary action taken, explictly switch off the others
    if args.probe or args.verify or args.bench:
        args.probe = first_defined(args.probe, False)
        args.verify = first_defined(args.verify, False)
        args.bench = first_defined(args.bench, False)

    # Build by default of required by a later step
    args.build = first_defined(
        args.build, args.probe or args.verify or args.bench, False)

    # Configure by default when building
    args.configure = first_defined(args.configure, args.build, False)

    # When benching, also evaluate that result by default
    args.evaluate = first_defined(args.evaluate, args.bench, False)


def determine_default_action(args):
    if args.bench and args.verify:
        return DefaultAction.VERIFY_THEN_BENCH
    if args.bench:
        return DefaultAction.BENCH
    if args.verify:
        return DefaultAction.VERIFY
    if args.probe and args.tune:
        return DefaultAction.PROBE_TUNED
    if args.probe:
        return DefaultAction.PROBE
    if args.tune:
        return DefaultAction.TUNE
    if args.build:
        return DefaultAction.BUILD
    if args.configure:
        return DefaultAction.CONFIGURE
    if args.clean:
        return DefaultAction.CLEAN
    if args.evaluate:
        return DefaultAction.EVALUATE
    if args.report or args.reportfile:
        return DefaultAction.REPORT
    return DefaultAction.VERIFY_THEN_BENCH


def apply_default_action(default_action, args):
    # Primary actions (execute the compiled program)
    # In some sense probing is not primary (done for it's own sake), but for preparation of a bench run
    args.bench = first_defined(args.bench, default_action in {DefaultAction.BENCH, DefaultAction.VERIFY_THEN_BENCH})
    args.probe = first_defined(args.probe, default_action in {DefaultAction.PROBE})
    args.tune = first_defined(args.tune, default_action in {DefaultAction.TUNE, DefaultAction.PROBE_TUNED})
    args.verify = first_defined(args.verify, default_action in {DefaultAction.VERIFY, DefaultAction.VERIFY_THEN_BENCH})

    # Auxiliary actions (those that are needed to do some primary action)
    if args.build is None:
        if default_action in {DefaultAction.BUILD}:
            args.build = True
        elif args.bench or args.probe or args.tune or args.verify:
            args.build = True
        else:
            args.build = False
    if args.configure is None:
        if default_action in {DefaultAction.CONFIGURE}:
            args.configure = True
        elif args.build:
            # Default: configure if not yet configured or args changed
            args.configure = None
        else:
            # Do not configure when not building
            args.configure = False
    args.clean = first_defined(args.clean, default_action in {DefaultAction.CLEAN})

    # Analysis actions
    args.evaluate = first_defined(
        args.evaluate,
        default_action in {
            DefaultAction.EVALUATE,
            DefaultAction.BENCH,
            DefaultAction.VERIFY_THEN_BENCH})
    args.report = first_defined(
        args.report,
        bool(
            args.reportfile) or None,
        default_action in {
            DefaultAction.BENCH,
            DefaultAction.VERIFY_THEN_BENCH,
            DefaultAction.REPORT})


verbose = None


class DriverMode:
    """Multiple builddirs managed by rosetta.py. May create+deleta builddirs as needed."""
    MANAGEDBUILDDIR = NamedSentinel('managed builddir(s)')

    """Running from a user-created cmake builddir. Must not change any CMakeCache settings. Also, only one config (Debug, Release, Minsize, ...) available"""
    USERBUILDDIR = NamedSentinel('user builddir')

    """Invoked by cmake/make itself (e.g. `ninja bench`). Should not trigger any re=configure or build, assume this has already been done."""
    FROMCMAKE = NamedSentinel('called by cmake')


class DefaultAction:
    CLEAN = NamedSentinel('clean')
    CONFIGURE = NamedSentinel('configure')
    BUILD = NamedSentinel('build')
    # TODO: Measure the compile/build time, so clean everytime
    BUILD_TIMED = NamedSentinel('timed_build')
    PROBE = NamedSentinel('probe')
    TUNE = NamedSentinel('tune')
    # Before probing a problem size, tune the benchmark with that problem size
    PROBE_TUNED = NamedSentinel('probe_tuned')
    VERIFY = NamedSentinel('verify')
    BENCH = NamedSentinel('bench')
    VERIFY_THEN_BENCH = NamedSentinel('verify_then_bench')
    EVALUATE = NamedSentinel('evaluate')
    REPORT = NamedSentinel('report')
    COMPARE = NamedSentinel('compare')


def driver_main(
        mode: DriverMode,
        argv: typing.List[str] = [None],
        default_action: DefaultAction = None,
        benchlistfile: pathlib.Path = None,
        srcdir: pathlib.Path = None,
        builddir: pathlib.Path = None,
        rootdir: pathlib.Path = None):
    assert mode in {DriverMode.USERBUILDDIR, DriverMode.MANAGEDBUILDDIR}
    assert default_action in {None, DefaultAction. CLEAN, DefaultAction. CONFIGURE, DefaultAction. BUILD,
                              DefaultAction.PROBE, DefaultAction.VERIFY, DefaultAction.BENCH, DefaultAction.EVALUATE}

    probestages = ['hybrid', 'runtime', 'compiletime'] if DriverMode.MANAGEDBUILDDIR else ['runtime']

    # TODO: Description according default_action
    parser = argparse.ArgumentParser(
        description="Benchmark configure, build, execute & evaluate", allow_abbrev=False)
    parser.add_argument('--verbose', '-v', action='count')
    
    # Maintainance
    if mode == DriverMode.MANAGEDBUILDDIR:
        parser.add_argument('--update-format', action='store_true', help="Reformat Rosetta's source files using clang-format, cmake-format and autopep8")
        parser.add_argument('--check-format', action='store_true', help="Return with error if --update-format would change any files")


    if mode == DriverMode.MANAGEDBUILDDIR:
        # Used by launcher which is itself in the repository root directory
        parser.add_argument('--rootdir', type=pathlib.Path,
                            default=rootdir, help=argparse.SUPPRESS)

    # Clean step (before configure if managed builddir)
    add_boolean_argument(parser, 'clean', default=False,
                         help="Start from scratch")

    if mode == DriverMode.MANAGEDBUILDDIR:
        # Configure step
        add_boolean_argument(parser, 'configure', default=None,
                             help="Enable configure (CMake) step (Default: if necessary)")
        # TODO: Add switches that parse multiple arguments using shsplit
        parser.add_argument('--config', metavar="CONFIG", action='append',
                            help="Configuration selection (must exist from previous invocations)", default=None)
        parser.add_argument('--ppm', metavar="CONFIG:PPM", action='append')
        parser.add_argument(
            '--cmake-arg', metavar="CONFIG:ARG", action='append')
        parser.add_argument(
            '--cmake-def', metavar="CONFIG:DEF[=VAL]", action='append')
        parser.add_argument(
            '--compiler-arg', metavar="CONFIG:ARG", action='append')
        parser.add_argument(
            '--compiler-def', metavar="CONFIG:DEF[=VAL]", action='append')

    # Build step
    add_boolean_argument(parser, 'build', default=None,
                         help="Enable build step")
    add_boolean_argument(parser, 'buildondemand', default=True,
                         help="build to ensure executables are up-to-date")

    # Probe step
    add_boolean_argument(parser, 'probe', default=None, help="Enable probing")
    parser.add_argument('--problemsizefile-out', type=pathlib.Path)
    parser.add_argument('--limit-walltime', type=parse_time)
    parser.add_argument('--limit-rss', type=parse_memsize)
    parser.add_argument('--limit-alloc', type=parse_memsize)
    parser.add_argument(
        '--probe-stage',
        choices=probestages,
        default=probestages[0],
        help="What toolchain stage to probe; compiletime allows the compiler to see the problem size")

    # Tune step
    add_boolean_argument(parser, 'tune', default=None,
                         help="Benchmark performance tuning")
    parser.add_argument(
        '--tune-stage',
        choices=probestages,
        default=probestages[0],
        help="What toolchain stage to tune; compiletime allows tuning source and compiler parameters")

    # Verify step
    add_boolean_argument(parser, 'verify', default=None,
                         help="Enable check step")

    # Benchmark step
    add_boolean_argument(parser, 'bench', default=None, help="Enable run step")
    parser.add_argument('--problemsizefile', type=pathlib.Path,
                        help="Problem sizes to use (.ini file)")  # Also used by --verify
    add_filter_args(parser)
    # Evaluate step
    add_boolean_argument(parser, 'evaluate', default=None,
                         help="Evaluate result")
    parser.add_argument(
        '--use-results-rdir',
        type=pathlib.Path,
        action='append',
        default=[],
        help="Use these result xml files from this dir (recursive); otherwise use result from benchmarking")
    parser.add_argument('--group-by', help="Comma-separated list of categories to group")
    parser.add_argument('--compare-by', help="Comma-separated list of categories to compare")
    parser.add_argument('--cols', help="Comma-separated list of columns to display (overrides auto selection)")
    parser.add_argument(
        '--cols-always',
        help="Comma-separated list of columns to display even if it does not contains data")
    parser.add_argument('--cols-never', help="Comma-separated list of columns to not display")
    # Report step
    add_boolean_argument(parser, 'report', default=None, help="Create HTML report")
    parser.add_argument('--reportfile', type=pathlib.Path, help="Path to write the report.html to")

    args = parser.parse_args(None if argv is None else [str(a) for a in argv[1:]])
    global verbose
    verbose = args.verbose

    level = log.INFO if args.verbose else log.WARNING
    log.basicConfig(level=level, format='{message}', style='{')
    try:
        import colorlog
        for h in log.root.handlers:
            h.setFormatter(colorlog.ColoredFormatter(
                '{log_color}{message}', style='{'))
    except BaseException:
        pass



    if mode == DriverMode.MANAGEDBUILDDIR:
        if args.update_format:
            from . import formatting
            return formatting.update_format(srcdir)
        elif args.check_format:
            from . import formatting
            return formatting.check_format(srcdir)
            


    if mode == DriverMode.USERBUILDDIR:
        args.configure = False

    main_action = default_action or determine_default_action(args)
    if not main_action:
        die("No action to take")
    apply_default_action(main_action, args)

    with globalctxmgr:
        resultfiles = None
        resultssubdir = None
        default_compare_by = None

        resultsdir = None

        def get_resultsdir():
            nonlocal resultsdir
            if not resultsdir:
                if mode == DriverMode.MANAGEDBUILDDIR:
                    resultsdir = rootdir / 'results'
                else:
                    resultsdir = builddir / 'results'
                resultsdir.mkdir(parents=True, exist_ok=True)
            return resultsdir

        if mode == DriverMode.MANAGEDBUILDDIR:
            # Always also create a REF dir in case we need it when running --verify
            configs = parse_build_configs(args, implicit_reference=True)
            if len(configs) >= 3: # Not counting REF
                default_compare_by = ['configname']

            rootdir = mkpath(first_defined(args.rootdir, pathlib.Path.cwd()))
            builddir = rootdir / 'build'

            # if builddir.exists():
            #    if args.clean:
            #        # TODO: Do this automatically when necessary (hash the CMakeLists.txt)
            #        print_verbose("Cleaning previous builds")
            #        for c in builddir.iterdir():
            #            if c.name == 'results':
            #                continue
            #            shutil.rmtree(c)
            #    else:
            #        print_verbose("Reusing existing build")

            # If only cleaning, clean everything
            # TODO: If specific config specified, clean only those
            if main_action == DefaultAction.CLEAN:
                for b in builddir.iterdir():
                    shutil.rmtree(b)
                return

            for config in configs:
                if not config.name:
                    config.builddir = builddir / 'defaultbuild'
                elif config.name == "REF":
                    # TODO: same as defaultbuild?
                    config.builddir = builddir / 'refbuild'
                else:
                    config.builddir = builddir / f'build-{config.name}'

            # Configure step
            # TODO: Recognize "module" system
            # TODO: Recognize famous environments (JLSE, Theta, Summit, Frontier, Aurora, ...)
            for config in configs:
                builddir = config.builddir
                configdescfile = builddir / 'RosettaCache.txt'

                if args.configure is not False:
                    opts = ['cmake', '-S', srcdir, '-B', builddir, '-GNinja Multi-Config',
                            '-DCMAKE_CROSS_CONFIGS=all', f'-DROSETTA_RESULTS_DIR={get_resultsdir()}']
                    opts += config.gen_cmake_args()
                    expectedopts = shjoin(opts).rstrip()

                    reusebuilddir = False
                    if config.is_predefined:
                        # Without cmake options, reuse whatever is in the builddir if it is already configured
                        reusebuilddir = (builddir / 'build.ninja').exists()
                    else:
                        if args.configure or args.clean:
                            pass
                        elif args.configure is False:
                            reusebuilddir = True
                        elif config.usecur and (builddir / 'build.ninja').exists():
                            reusebuilddir = True
                        elif configdescfile.is_file() and (builddir / 'build.ninja').exists():
                            existingopts = readfile(configdescfile).rstrip()
                            if existingopts == expectedopts:
                                reusebuilddir = True

                    if not reusebuilddir:
                        if builddir.exists() and args.clean:
                            shutil.rmtree(builddir)
                        if args.configure or (args.configure is None and (args.build or args.verify or args.probe)):
                            # TODO: Support other generators as well
                            builddir.mkdir(exist_ok=True, parents=True)
                            invoke_verbose(*opts, cwd=config.builddir)
                            createfile(configdescfile, expectedopts)

            if args.build:
                for config in configs:
                    # TODO: Select subset to be build
                    invoke_verbose('ninja', cwd=config.builddir)

            # Load all available benchmarks
            if args.verify or args.bench or args.probe:
                for config in configs:
                    if config.name == 'REF':
                        continue
                    registry.load_register_file(config.builddir / 'benchmarks' / 'benchlist.py')

            def only_REF(bench):
                return bench.configname == 'REF'

            def no_ref(bench):
                return bench.configname != 'REF' and bench.ppm in config.ppm
            if args.bench or args.verify or args.probe:
                refconfigs = list(c for c in configs if c.name == 'REF')
                refconfig = None
                if len(refconfigs) == 1:
                    [refconfig] = refconfigs
        else:
            configure_uptodate = False

            if args.build:
                # TODO: Build only filtered executables
                invoke_verbose('cmake', '--build', builddir, cwd=builddir)
                configure_uptodate = True

            if main_action == DefaultAction.BUILD:
                return

            if not configure_uptodate:
                invoke_verbose('cmake', '--build', builddir,
                               '--target', 'build.ninja', cwd=builddir)

            # Discover available benchmarks
            registry.load_register_file(benchlistfile)

        


        if args.probe:
            from . import prober
            assert args.problemsizefile_out, "Requires to set a problemsizefile to set"
            prober.run_probe(problemsizefile=args.problemsizefile_out, limit_walltime=args.limit_walltime,
                             limit_rss=args.limit_rss, limit_alloc=args.limit_alloc)

        if args.verify:
            from . import verifier
            if mode == DriverMode.MANAGEDBUILDDIR:
                refdir = (refconfig.builddir if refconfig else None) / 'refout'
            else:
                refdir = builddir / 'refout'
            verifier.run_verify(problemsizefile=args.problemsizefile, refdir=refdir)

        if args.bench:
            from . import runner
            resultfiles, resultssubdir = runner.run_bench(
                srcdir=srcdir, problemsizefile=args.problemsizefile, resultdir=get_resultsdir(), args=args)

        if args.evaluate or args.report:
            from . import evaluator

            for use_results_rdir in args.use_results_rdir:
                resultfiles = resultfiles or []
                resultfiles += mkpath(use_results_rdir).glob('**/*.xml')
            resultfiles = sorted(resultfiles)

            # If there is no other source of results, source all previous ones
            if resultfiles is None:
                resultfiles = resultfiles or []
                if resultsdir:
                    for xmlfile in resultsdir .rglob("*.xml"):
                        resultfiles.append(xmlfile)

            if resultfiles is None:
                die("No source for resultfiles")
            if not resultfiles:
                die("No results")

            results = evaluator.load_resultfiles(resultfiles)
            # if not match_filter(e, args):
            #     continue

            if args.evaluate:
                group_by = None
                if args.group_by is not None:
                    group_by = [s.strip() for s in args.group_by.split(',')]
                compare_by = default_compare_by
                if args.compare_by is not None:
                    compare_by = [s.strip() for s in args.compare_by.split(',')]
                cols = None
                if args.cols is not None:
                    cols = [s.trim() for s in args.cols.split('')]
                cols_always = [s.trim() for s in args.cols_always.split('')] if args.cols_always else ['program']
                cols_never = [s.trim() for s in args.cols_never.split('')] if args.cols_never else []
                evaluator.results_compare(
                    results,
                    compare_by=compare_by,
                    group_by=group_by,
                    always_columns=cols_always,
                    never_columns=cols_never,
                    columns=cols)

            if args.report:
                if args.reportfile:
                    reportfile = args.reportfile
                elif resultssubdir:
                    # If emitting a report the analyze the last benchmark run, put the report into that directory
                    reportfile = resultssubdir / 'report.html'
                else:
                    now = datetime.datetime.now()  # TODO: Use runner.make_resultssubdir
                    reportfile = mkpath(f"report_{now:%Y%m%d_%H%M}.html")
                    if resultsdir := get_resultsdir():
                        reportfile = resultsdir / reportfile
                evaluator.save_report(results, filename=reportfile)
