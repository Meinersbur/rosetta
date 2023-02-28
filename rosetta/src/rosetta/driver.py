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
import pathlib
import subprocess
import argparse
import sys
from itertools import count
from cmath import exp
from .util.cmdtool import *
from .util.orderedset import OrderedSet
from .util.support import *
from .util import invoke
from .common import *
from .evaluator import subcommand_evaluate
import re
import logging as log
import typing
from . import runner
from .builder import *


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

    keys = OrderedSet()
    if implicit_reference:
        keys.add("REF")
    keys |= cmake_arg.keys()
    keys |= cmake_def.keys()
    keys |= compiler_arg.keys()
    keys |= compiler_def.keys()
    keys |= ppm.keys()

    configs = []
    for k in keys:
        if not k:
            continue
        # TODO: Handle duplicate defs (specific override general)
        configs.append(BuildConfig(name=k, ppm=ppm[''] + ppm[k], cmake_arg=cmake_arg[''] + cmake_arg[k], cmake_def=cmake_def[''] |
                       cmake_def[k], compiler_arg=compiler_arg[''] + compiler_arg[k], compiler_def=compiler_def[''] | compiler_def[k]))

    # Add additional configurations that are stored in RosettaCache.txt files.
    for k in first_defined(args.config, []):
        if k not in keys:
            configs.append(BuildConfig(name=k, is_predefined=True, ppm=None,
                           cmake_arg=None, cmake_def=None, compiler_arg=None, compiler_def=None))

    # Use single config if no "CONFIG:" is specified
    if not configs:
        configs.append(BuildConfig(name=None, ppm=ppm[''], cmake_arg=cmake_arg[''], cmake_def=cmake_def[''],
                       compiler_arg=compiler_arg[''], compiler_def=compiler_def['']))

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
    if args.probe:
        return DefaultAction.PROBE
    if args.build:
        return DefaultAction.BUILD
    if args.configure:
        return DefaultAction.CONFIGURE
    if args.clean:
        return DefaultAction.CLEAN
    if args.evaluate:
        return DefaultAction.EVALUATE
    return DefaultAction.VERIFY_THEN_BENCH


def apply_default_action(default_action, args):
    args.clean = first_defined(args.clean, default_action in {
                               DefaultAction.CLEAN} or None)
    args.configure = first_defined(args.configure, default_action in {
                                   DefaultAction.CONFIGURE} or None)
    args.build = first_defined(args.build, default_action in {
                               DefaultAction.BUILD,       DefaultAction.BENCH, DefaultAction.VERIFY, DefaultAction.VERIFY_THEN_BENCH })
    args.probe = first_defined(
        args.probe, default_action in {DefaultAction.PROBE})
    args.verify = first_defined(args.verify, default_action in {
                                DefaultAction.VERIFY, DefaultAction.VERIFY_THEN_BENCH})
    args.bench = first_defined(args.bench, default_action in {
                               DefaultAction.BENCH, DefaultAction.VERIFY_THEN_BENCH})
    args.evaluate = first_defined(args.evaluate, default_action in {
                                  DefaultAction.EVALUATE})
    args.report = first_defined(args.evaluate, default_action in { DefaultAction.BENCH, DefaultAction.REPORT})
    args.compare = first_defined(args.compare, default_action in { DefaultAction.COMPARE})


verbose = None



class DriverMode:
    MANAGEDBUILDDIR = NamedSentinel('managed builddir(s)')
    USERBUILDDIR = NamedSentinel('user builddir')
    FROMCMAKE = NamedSentinel('called by cmake')



class DefaultAction:
    CLEAN = NamedSentinel('clean')
    CONFIGURE = NamedSentinel('configure')
    BUILD = NamedSentinel('build')
    PROBE = NamedSentinel('probe')
    VERIFY = NamedSentinel('verify')
    BENCH = NamedSentinel('bench')
    VERIFY_THEN_BENCH = NamedSentinel('verify_then_bench')
    EVALUATE = NamedSentinel('evaluate')
    REPORT = NamedSentinel('report')
    COMPARE = NamedSentinel('compare')


def driver_main(
        argv: typing.List[str],
        mode: DriverMode,
        default_action: DefaultAction = None,
        benchlistfile:pathlib.Path = None,
        srcdir:pathlib.Path=None,
        builddir:pathlib.Path=None,
        rootdir:pathlib.Path=None):
    assert argv is not None
    assert mode in {DriverMode.USERBUILDDIR, DriverMode.MANAGEDBUILDDIR}
    assert default_action in {None, DefaultAction. CLEAN, DefaultAction. CONFIGURE, DefaultAction. BUILD, DefaultAction.PROBE, DefaultAction.VERIFY, DefaultAction.BENCH, DefaultAction.EVALUATE}

    if mode == DriverMode.USERBUILDDIR:
        assert benchlistfile is not None
        assert rootdir is None
        assert builddir is not None


    if mode == DriverMode.MANAGEDBUILDDIR:
        assert benchlistfile is  None
        assert builddir is None

    # TODO: Description according default_action
    parser = argparse.ArgumentParser(
        description="Benchmark configure, build, execute & evaluate", allow_abbrev=False)
    parser.add_argument('--verbose', '-v', action='count')

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
    parser.add_argument('--limit-walltime', type=parse_time)
    parser.add_argument('--limit-rss', type=parse_memsize)
    parser.add_argument('--limit-alloc', type=parse_memsize)

    # Verify step
    add_boolean_argument(parser, 'verify', default=None,
                         help="Enable check step")

    # Benchmark step
    add_boolean_argument(parser, 'bench', default=None, help="Enable run step")
    parser.add_argument('--problemsizefile', type=pathlib.Path,
                        help="Problem sizes to use (.ini file)")  # Also used by --verify

    # Evaluate step
    add_boolean_argument(parser, 'evaluate', default=None,
                         help="Evaluate result")
    parser.add_argument('--boxplot', type=pathlib.Path,
                        metavar="FILENAME", help="Save as boxplot to FILENAME")

    # Report step
    add_boolean_argument(parser, 'report', default=None, help="Create HTML report")

    # Compare step
    add_boolean_argument(parser, 'compare', default=None, help="Compare two or more benchmark runs")


    args = parser.parse_args(argv[1:])
    global verbose
    verbose = args.verbose

    level = log.INFO if args.verbose else log.WARNING
    log.basicConfig(level=level, format='{message}', style='{')
    try:
        import colorlog
        for h in log.root.handlers:
            h.setFormatter(colorlog.ColoredFormatter(
                '{log_color}{message}', style='{'))
    except:
        pass

    if mode == DriverMode.USERBUILDDIR:
        args.configure = False

    main_action = default_action or determine_default_action(args)
    if not main_action:
        die("No action to take")
    apply_default_action(main_action, args)

    with globalctxmgr:
        if mode == DriverMode.MANAGEDBUILDDIR:
            configs = parse_build_configs(args, implicit_reference=args.verify)

            rootdir = mkpath(first_defined(args.rootdir, pathlib.Path.cwd()))
            builddir = rootdir / 'build'
            resultdir = rootdir / 'results'



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
            resultdir.mkdir(parents=True, exist_ok=True)

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

                    # TODO: Support other generators as well
                    opts = ['cmake', '-S', srcdir,  '-B', builddir, '-GNinja Multi-Config',
                            '-DCMAKE_CROSS_CONFIGS=all', f'-DROSETTA_RESULTS_DIR={resultdir}']
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
                        elif configdescfile.is_file() and (builddir / 'build.ninja').exists():
                            existingopts = readfile(configdescfile).rstrip()
                            if existingopts == expectedopts:
                                reusebuilddir = True

                    if not reusebuilddir:
                        if builddir.exists() and args.clean:
                            shutil.rmtree(builddir)
                        if args.configure or (args.configure is  None and (args.build or args.verify or args.probe)):
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
                    runner.load_register_file(
                        config.builddir / 'benchmarks' / 'benchlist.py')

            def only_REF(bench):
                return bench.configname == 'REF'

            def no_ref(bench):
                return bench.configname != 'REF' and bench.ppm in config.ppm
            if args.bench:
                try:
                    [refconfig] = (c for c in configs if c.name == 'REF')
                except BaseException:
                    refconfig = None
                resultfiles = runner.subcommand_run(None, args,
                                                    srcdir=rootdir,
                                                    buildondemand=not args.build,
                                                    builddirs=[
                                                        config.builddir for config in configs],
                                                    refbuilddir=refconfig.builddir if refconfig else None,
                                                    filterfunc=no_ref,
                                                    resultdir=resultdir)
            else:
                # If not evaluating the just-executed, search for previously saved result files.

                # TODO: Filter result files
                resultfiles = []
                for xmlfile in resultdir .rglob("*.xml"):
                    resultfiles.append(xmlfile)

            subcommand_evaluate(
                None, args, resultfiles=resultfiles, resultsdir=resultdir)



        if mode == DriverMode.USERBUILDDIR:
            # If neither no action is specified, enable --bench implicitly unless --no-bench
            probe = args.probe
            verify = args.verify
            bench = args.bench
            # if bench is None and not verify and not probe:
            #    bench = True

            resultdir = builddir / 'results'
            configure_uptodate =False

            if args.build:
                # TODO: Build only filtered executables
                invoke_verbose('cmake', '--build', builddir, cwd=builddir)
                configure_uptodate=True

            if main_action == DefaultAction.BUILD:
                return 


            
            if not configure_uptodate:
                invoke_verbose('cmake', '--build',builddir,  '--target', 'build.ninja',  cwd=builddir)

            # Discover available benchmarks
            runner.load_register_file(benchlistfile)

            if probe:
                assert args.problemsizefile, "Requires to set a problemsizefile to set"
                run_probe(problemsizefile=args.problemsizefile, limit_walltime=args.limit_walltime, limit_rss=args.limit_rss, limit_alloc=args.limit_alloc)

            if verify:
                refdir = refbuilddir / 'refout'
                run_verify(problemsizefile=args.problemsizefile, refdir=refdir)

            resultfiles = None
            if bench:
                resultfiles = runner.run_bench(srcdir=srcdir, problemsizefile=args.problemsizefile, resultdir=resultdir)

            return resultfiles


