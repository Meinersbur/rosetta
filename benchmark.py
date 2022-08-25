#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from collections import defaultdict
import sys
import argparse
import pathlib
from pathlib import Path
import shutil
import re
import importlib



# TODO: There must be a better way, including for vscode to find it
# TODO: move all the plumbing into lib
if __name__ == '__main__':
    #  FIXME: hardcoded path
    sys.path.insert(0, '/home/meinersbur/src/rosetta/rosetta/lib') 
import rosetta
runner = rosetta.runner

script = Path(sys.argv[0]).absolute()
thisscript = Path(__file__)
thisscriptdir = thisscript.parent

sys.path.insert(0,str( (thisscript.parent / 'rosetta' /  'lib').absolute() ))
from rosetta import *




class BuildConfig:
    def __init__(self,name,ppm,cmake_arg,cmake_def,compiler_arg,compiler_def):
        self.name = name
        self.ppm = set(ppm)
        self.cmake_arg = cmake_arg
        self.cmake_def = cmake_def
        self.compiler_arg = compiler_arg 
        self.compiler_def = compiler_def

        # TODO: select compiler executable

    def gen_cmake_args(self):
        compiler_args = self.compiler_arg.copy()
        for k,v in self.compiler_def:
            if v:
                compiler_args.append(f"-D{k}")
            else:
                compiler_args.append(f"-D{k}={v}")

        # TODO: Combine with (-D, -DCMAKE_<lang>_FLAGS) from compiler/cmake_arg
        cmake_opts = self.cmake_arg[:]
        for k,d in self.cmake_def:
            cmake_opts .append(f"-D{k}={d}")
        if compiler_args:
            # TODO: Only set the ones relevant for enable PPMs
            opt_args  = shjoin(compiler_args)
            cmake_opts += [f"-DCMAKE_C_FLAGS={opt_args}", f"-DCMAKE_CXX_FLAGS={opt_args}", f"-DCMAKE_CUDA_FLAGS={opt_args}"] # TODO: Release flags?

        if self.ppm:
          # TODO: Case, shortcuts  
          for ppm in ['serial', 'cuda', 'openmp-thread', 'openmp-task', 'openmp-target']:
            ucase_name = ppm.upper().replace('-','_')
            if ppm in self.ppm:
                cmake_opts.append(f"-DROSETTA_PPM_{ucase_name}=ON")
            else:
                # TODO: switch to have default OFF, so we don't need to list all of them
                cmake_opts.append(f"-DROSETTA_PPM_{ucase_name}=OFF")

        if self.name:
            cmake_opts.append(f"-DROSETTA_CONFIGNAME={self.name}")

        return cmake_opts


def make_buildconfig(name,ppm,cmake_arg,cmake_def,compiler_arg,compiler_def):
    return BuildConfig(name,ppm,cmake_arg,cmake_def,compiler_arg,compiler_def)


configsplitarg =         re.compile(r'((?P<configname>[a-zA-Z0-9_]+)\:)?(?P<arg>.*)')
configsplitdef =         re.compile(r'((?P<configname>[a-zA-Z0-9_]+)\:)?(?P<defname>[a-zA-Z0-9_]+)(\=(?P<defvalue>.*))?')
configsplitdefrequired = re.compile(r'((?P<configname>[a-zA-Z0-9_]+)\:)?(?P<defname>[a-zA-Z0-9_]+)\=(?P<defvalue>.*)')
configppm =         re.compile(r'((?P<configname>[a-zA-Z0-9_]+)\:)?(?P<ppm>[a-zA-Z\-]+)')
def parse_build_configs(args,implicit_reference):
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
    def parse_deflists(l,valrequired):
        raw = defaultdict(lambda: dict())
        if l is None:
            return raw
        for arg in l:
            m = (configsplitdefrequired if valrequired else configsplitdefrequired).fullmatch(arg)
            configname = m.group('configname') or ''
            defname = m.group('defname')
            defvalue = m.group('defvalue') # Can be none
            raw[configname][defname] = defvalue
        return raw

    cmake_arg = parse_arglists(args.cmake_arg)
    cmake_def = parse_deflists(args.cmake_def,valrequired=True)
    compiler_arg = parse_arglists(args.compiler_arg)
    compiler_def = parse_deflists(args.compiler_def,valrequired=False)
    ppm = parse_arglists(args.ppm)

    keys = OrderedSet()
    if implicit_reference:
        keys.add("REF")
    keys |= cmake_arg.keys()
    keys |= cmake_def.keys()
    keys |= compiler_arg.keys()
    keys |= compiler_def.keys()


    configs = []
    for k in keys:
        if not k:
            continue
        # TODO: Handle duplicate defs (specific override general)
        configs.append(BuildConfig(k, ppm[''] +  ppm[k],  cmake_arg=cmake_arg[''] + cmake_arg[k] ,  cmake_def=cmake_def[''] | cmake_def[k],  compiler_arg=compiler_arg[''] + compiler_arg[k], compiler_def=compiler_def[''] | compiler_def[k] ))
    # Use single config if not "CONFIG:" is specified
    if not configs:
       configs.append(BuildConfig(None, ppm[''], cmake_arg=cmake_arg[''] ,  cmake_def=cmake_def[''],  compiler_arg=compiler_arg[''] , compiler_def=compiler_def[''] ) )
    return configs

def print_verbose(*args, **kwargs):
    if verbose:
        print(*args, **kwargs)

def invoke_verbose(*args, **kwargs):
    if verbose :
        invoke.diag(*args, **kwargs)
    else:
        invoke.run(*args, **kwargs)


def main(argv):
    global verbose
    parser = argparse.ArgumentParser(description="Benchmark configure, build, execute & evaluate", allow_abbrev=False)

    # Pipeline actions
    add_boolean_argument(parser, 'clean', default=False, help="Start from scratch")
    add_boolean_argument(parser, 'configure', default=True, help="Enable configure (CMake) step")
    add_boolean_argument(parser, 'build', default=True, help="Enable build step")
    add_boolean_argument(parser, 'verify', default=True, help="Enable check step")
    add_boolean_argument(parser, 'run', default=True, help="Enable run step")
    add_boolean_argument(parser, 'evaluate', default=True, help="Print results table")

    # TODO: Warn/error on unused arguments because action is disable
    parser.add_argument('--problemsizefile', type=pathlib.Path, help="Problem sizes to use (.ini file)")
    parser.add_argument('--verbose', '-v', action='count')
    

    #TODO: Add switches that parse multiple arguments using shsplit
    parser.add_argument('--cmake-arg', metavar="CONFIG:ARG", action='append')
    parser.add_argument('--cmake-def', metavar="CONFIG:DEF[=VAL]", action='append')
    parser.add_argument('--compiler-arg', metavar="CONFIG:ARG", action='append')
    parser.add_argument('--compiler-def', metavar="CONFIG:DEF[=VAL]", action='append')
    parser.add_argument('--ppm', metavar="CONFIG:PPM", action='append')

    parser.add_argument('--boxplot', type=pathlib.Path)

    

    args = parser.parse_args(argv[1:])
    verbose = args.verbose

    # TODO: If not specified, just reuse existing configs 
    configs = parse_build_configs(args,implicit_reference=args.verify)

    srcdir = script.parent
    builddir = srcdir / 'build'
    resultdir = builddir / 'results'

    #if builddir.exists():
    #    if args.clean:
    #        # TODO: Do this automatically when necessary (hash the CMakeLists.txt)
    #        print_verbose("Cleaning previous builds")
    #        for c in builddir.iterdir():
    #            if c.name == 'results':
    #                continue
    #            shutil.rmtree(c)
    #    else:
    #        print_verbose("Reusing existing build")
    resultdir.mkdir(parents=True,exist_ok=True)

    
    for config in configs:
        if not config.name:
              config.builddir  = builddir / 'defaultbuild'
        elif config.name == "REF":
            # TODO: same as defaultbuild?
            config.builddir  = builddir / 'refbuild'
        else:
            config.builddir = builddir / f'build-{config.name}'


    # TODO: Recognize "module" system
    # TODO: Recognize famous machines (JLSE, Summit, Aurora, Frontier, ...)


    for config in configs:
        builddir = config.builddir 
        configdescfile  = builddir/'RosettaCache.txt'

        # TODO: Support other generators as well
        opts = ['cmake', srcdir, '-GNinja Multi-Config', '-DCMAKE_CROSS_CONFIGS=all', f'-DROSETTA_RESULTS_DIR={resultdir}']
        opts += config.gen_cmake_args()
        expectedopts = shjoin(opts)

        reusebuilddir = False
        if not args.clean and configdescfile.is_file() and (builddir / 'build.ninja').exists():
            existingopts = readfile(configdescfile)            
            if existingopts == expectedopts:
                reusebuilddir=True

        if not reusebuilddir:
            if builddir.exists():
                shutil.rmtree(builddir)
            builddir.mkdir(exist_ok=True,parents=True)
            invoke_verbose(*opts, cwd=config.builddir)
            createfile(configdescfile, expectedopts)
        

          
            
            

        if args.build:
            # TODO: Select subset to be build 
            invoke_verbose('ninja', cwd=config.builddir)


    # Load all available benchmarks
    if args.verify or args.run:
        for config in configs:
            runconfigfile = config.builddir / 'run-Release.py' # TODO: Always Release?
            spec = importlib.util.spec_from_file_location(str(runconfigfile), str(runconfigfile))
            module =  importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)


    if args.verify:
       def only_REF(bench):
           return bench.configname =='REF' 
       [refconfig] = (c for c in configs if c.name == 'REF')
       refdir = refconfig.builddir / 'refout'
       refdir.mkdir(exist_ok=True,parents=True)
       runner.ensure_reffiles(problemsizefile=args.problemsizefile,filterfunc=only_REF,srcdir=srcdir,refdir=refdir)

       runner.run_verify(problemsizefile=args.problemsizefile,filterfunc=only_REF,srcdir=srcdir,refdir=refdir)


    resultfiles = None
    if args.run:
        # TODO: Filter way 'REF' config
        resultfiles = runner.run_bench(srcdir=thisscriptdir, problemsizefile=args.problemsizefile)
        #invoke_verbose('cmake', '--build', '.',  '--config','Release', '--target','run', cwd=config.builddir)
    

    if args.evaluate:
        if not resultfiles:
            assert False, "TODO: Lookup last (successful) results dir"
        if len(configs) == 1:
            runner.evaluate(resultfiles)
        else:
            runner.results_compare(resultfiles, compare_by="configname", compare_val=["walltime"])

    if args.boxplot:
        def no_ref(bench):
           return bench.configname !='REF' 
        fig = runner.results_boxplot(resultfiles, compare_by="configname", filterfunc=no_ref)
        fig.savefig(fname=args.boxplot)
        fig.canvas.draw_idle() 




if __name__ == '__main__':
    retcode = main(argv=sys.argv)
    if retcode:
        exit(retcode)


