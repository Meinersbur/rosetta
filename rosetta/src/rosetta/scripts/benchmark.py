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
import  rosetta.util.invoke as invoke


script = Path(sys.argv[0]).absolute()
thisscript = Path(__file__)
thisscriptdir = thisscript.parent




class BuildConfig:
    def __init__(self, name, ppm, cmake_arg, cmake_def, compiler_arg, compiler_def, is_predefined=False):
        self.name = name
        self.ppm = set(ppm)
        self.cmake_arg = cmake_arg
        self.cmake_def = cmake_def
        self.compiler_arg = compiler_arg
        self.compiler_def = compiler_def
        self.is_predefined = is_predefined

        # TODO: select compiler executable

    def gen_cmake_args(self):
        compiler_args = self.compiler_arg.copy()
        for k, v in self.compiler_def:
            if v:
                compiler_args.append(f"-D{k}")
            else:
                compiler_args.append(f"-D{k}={v}")

        # TODO: Combine with (-D, -DCMAKE_<lang>_FLAGS) from compiler/cmake_arg
        cmake_opts = self.cmake_arg[:]
        for k, d in self.cmake_def.items():
            cmake_opts .append(f"-D{k}={d}")
        if compiler_args:
            # TODO: Only set the ones relevant for enable PPMs
            opt_args = shjoin(compiler_args)
            cmake_opts += [f"-DCMAKE_C_FLAGS={opt_args}", f"-DCMAKE_CXX_FLAGS={opt_args}",
                           f"-DCMAKE_CUDA_FLAGS={opt_args}"]  # TODO: Release flags?

        if self.ppm:
            # TODO: Case, shortcuts
            for ppm in ['serial', 'cuda', 'openmp-thread', 'openmp-task', 'openmp-target']:
                ucase_name = ppm.upper().replace('-', '_')
                if ppm in self.ppm:
                    cmake_opts.append(f"-DROSETTA_PPM_{ucase_name}=ON")
                else:
                    # TODO: switch to have default OFF, so we don't need to list all of them
                    cmake_opts.append(f"-DROSETTA_PPM_{ucase_name}=OFF")

        if self.name:
            cmake_opts.append(f"-DROSETTA_CONFIGNAME={self.name}")

        return cmake_opts


def make_buildconfig(name, ppm, cmake_arg, cmake_def, compiler_arg, compiler_def):
    return BuildConfig(name, ppm, cmake_arg, cmake_def, compiler_arg, compiler_def)


configsplitarg = re.compile(r'((?P<configname>[a-zA-Z0-9_]+)\:)?(?P<arg>.*)')
configsplitdef = re.compile(r'((?P<configname>[a-zA-Z0-9_]+)\:)?(?P<defname>[a-zA-Z0-9_]+)(\=(?P<defvalue>.*))?')
configsplitdefrequired = re.compile(r'((?P<configname>[a-zA-Z0-9_]+)\:)?(?P<defname>[a-zA-Z0-9_]+)\=(?P<defvalue>.*)')
configppm = re.compile(r'((?P<configname>[a-zA-Z0-9_]+)\:)?(?P<ppm>[a-zA-Z\-]+)')


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
            m = (configsplitdefrequired if valrequired else configsplitdefrequired).fullmatch(arg)
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
    for k in first_defined( args.config, []):
            if k not in keys:
                configs.append(BuildConfig(name=k,is_predefined=True, ppm=None,cmake_arg=None,cmake_def=None, compiler_arg=None,compiler_def=None))

    # Use single config if no "CONFIG:" is specified
    if not configs:
        configs.append(BuildConfig(ppm=ppm[''], cmake_arg=cmake_arg[''], cmake_def=cmake_def[''],
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


def main(argv, rootdir=None):
    global verbose
    parser = argparse.ArgumentParser(description="Benchmark configure, build, execute & evaluate", allow_abbrev=False)


    # Used by launcher which is itself in the repository root directory
    parser.add_argument('--rootdir', type=pathlib.Path, default=rootdir, help=argparse.SUPPRESS)

    # Clean step
    add_boolean_argument(parser, 'clean', default=False, help="Start from scratch")

    # Configure step
    add_boolean_argument(parser, 'configure', default=True, help="Enable configure (CMake) step")
    # TODO: Add switches that parse multiple arguments using shsplit
    parser.add_argument('--config', metavar="CONFIG", action='append', help="Configuration selection (must exist from previous invocations)",default=None)
    parser.add_argument('--ppm', metavar="CONFIG:PPM", action='append')
    parser.add_argument('--cmake-arg', metavar="CONFIG:ARG", action='append')
    parser.add_argument('--cmake-def', metavar="CONFIG:DEF[=VAL]", action='append')
    parser.add_argument('--compiler-arg', metavar="CONFIG:ARG", action='append')
    parser.add_argument('--compiler-def', metavar="CONFIG:DEF[=VAL]", action='append')

    # Build step
    add_boolean_argument(parser, 'build', default=True, help="Enable build step")

    # Run/verify/evaluate steps
    runner.subcommand_run(parser, None, srcdir=thisscriptdir)

    args = parser.parse_args(argv[1:])
    verbose = args.verbose

    with runner.globalctxmgr:

        # TODO: If not specified, just reuse existing configs
        configs = parse_build_configs(args, implicit_reference=args.verify)

        rootdir = mkpath(first_defined(args.rootdir,pathlib.Path.cwd()))
        builddir = rootdir / 'build'
        resultdir = builddir / 'results'

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

        for config in configs:
            if not config.name:
                config.builddir = builddir / 'defaultbuild'
            elif config.name == "REF":
                # TODO: same as defaultbuild?
                config.builddir = builddir / 'refbuild'
            else:
                config.builddir = builddir / f'build-{config.name}'

        # TODO: Recognize "module" system
        # TODO: Recognize famous environments (JLSE, Theta, Summit, Frontier, Aurora, ...)

        for config in configs:
            builddir = config.builddir
            configdescfile = builddir / 'RosettaCache.txt'

            # TODO: Support other generators as well
            opts = ['cmake', rootdir, '-GNinja Multi-Config', '-DCMAKE_CROSS_CONFIGS=all', f'-DROSETTA_RESULTS_DIR={resultdir}']
            opts += config.gen_cmake_args()
            expectedopts = shjoin(opts).rstrip()

            reusebuilddir = False
            if config.is_predefined:
              # Without cmake options, reuse whatever is in the builddir if it is already configured
              reusebuilddir = (builddir / 'build.ninja').exists()
            else:
              if not args.clean and configdescfile.is_file() and (builddir / 'build.ninja').exists():
                existingopts = readfile(configdescfile).rstrip()
                if existingopts == expectedopts:
                    reusebuilddir = True

            if not reusebuilddir:
                if builddir.exists():
                    shutil.rmtree(builddir)
                builddir.mkdir(exist_ok=True, parents=True)
                invoke_verbose(*opts, cwd=config.builddir)
                createfile(configdescfile, expectedopts)

        for config in configs:
            if args.build:
                # TODO: Select subset to be build
                invoke_verbose('ninja', cwd=config.builddir)

        # Load all available benchmarks
        if args.verify or args.bench or args.probe or (not args.verify and args.bench is None and not args.probe):
            for config in configs:
                runner.load_register_file(config.builddir / 'benchmarks' / 'benchlist.py')

        def only_REF(bench):
            return bench.configname == 'REF'

        def no_ref(bench):
            return bench.configname != 'REF'

        try:
            [refconfig] = (c for c in configs if c.name == 'REF')
        except BaseException:
            refconfig = None
        runner.subcommand_run(None, args,
                              srcdir=rootdir,
                              buildondemand=not args.build,
                              builddirs=[config.builddir for config in configs],
                              refbuilddir=refconfig.builddir if refconfig else None,
                              filterfunc=no_ref,
                              resultdir=resultdir
                              )

        # if args.verify:
        #   def only_REF(bench):
        #       return bench.configname =='REF'
        #   [refconfig] = (c for c in configs if c.name == 'REF')
        #   refdir = refconfig.builddir / 'refout'
        #   refdir.mkdir(exist_ok=True,parents=True)
        #   runner.ensure_reffiles(problemsizefile=args.problemsizefile,filterfunc=only_REF,srcdir=srcdir,refdir=refdir)
        #   runner.run_verify(problemsizefile=args.problemsizefile,filterfunc=only_REF,srcdir=srcdir,refdir=refdir)

        #resultfiles = None
        # if args.run:
        #    # TODO: Filter way 'REF' config
        #    resultfiles = runner.run_bench(srcdir=thisscriptdir, problemsizefile=args.problemsizefile)
        #    #invoke_verbose('cmake', '--build', '.',  '--config','Release', '--target','run', cwd=config.builddir)

        # if args.evaluate:
        #   if not resultfiles:
        #        assert False, "TODO: Lookup last (successful) results dir"
        #    if len(configs) == 1:
        #        runner.evaluate(resultfiles)
        #    else:
        #        runner.results_compare(resultfiles, compare_by="configname", compare_val=["walltime"])

        # if args.boxplot:
        #    def no_ref(bench):
        #       return bench.configname !='REF'
        #    fig = runner.results_boxplot(resultfiles, compare_by="configname", filterfunc=no_ref)
        #    fig.savefig(fname=args.boxplot)
        #    fig.canvas.draw_idle()


if __name__ == '__main__':
    retcode = main(argv=sys.argv)
    if retcode:
        exit(retcode)
