# -*- coding: utf-8 -*-

import sys
if not sys.version_info >= (3, 9):
    print("Requires python 3.9 or later", file=sys.stderr)
    sys.exit(1)


import argparse
import configparser
import importlib
import pathlib
import re
from io import StringIO
import importlib.util

from rosetta.util.support import *
import rosetta.runner as runner
import rosetta.registry as registry


buildre = re.compile(r'^\s*//\s*BUILD\:(?P<script>.*)$')
preparere = re.compile(r'^\s*//\s*PREPARE\:(?P<script>.*)$')


def compute_global_indent(slist):
    gindent = None
    for line in slist:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if gindent is None or indent < gindent:
            gindent = indent
    return gindent or 0


def unindent(slist, amount):
    for line in slist:
        yield line[amount:]


def global_unintent(slist):
    gindent = compute_global_indent(slist)
    yield from unindent(slist, gindent)


def gen_benchtargets(outfile, problemsizefile, benchdir, builddir, configname, filter=None):
    problemsizefile = runner.get_problemsizefile(
        problemsizefile=problemsizefile)
    config = configparser.ConfigParser()
    config.read(problemsizefile)

    buildfiles = []
    potentialbuildfiles = []
    for path in benchdir.rglob('*'):
        if not path.suffix.lower() in {'.cxx', '.cu', '.build'}:
            continue
        potentialbuildfiles.append(path)

        # Predict basename
        # TODO: Separate filter path/basename
        rel = path.parent.parent.relative_to(benchdir)
        basename = path.stem
        basename = '.'.join(list(rel.parts) + [basename])

        if filter and not any(f in basename for f in filter):
            #print(f"Benchmark {basename} does not match --filter-include={filter}")
            log.info(f"Benchmark {basename} does not match --filter-include={filter}")
            continue
        log.info(f"Adding benchmark {path}")
        buildfiles.append(path)

    benchs = []
    configdepfiles = []

    for buildfile in buildfiles:
        with buildfile.open() as f:
            script = []
            while True:
                line = f.readline()
                if not line:
                    break
                m = buildre.match(line)
                if m:
                    s = m.group('script')
                    script.append(s)
            if script:
                configdepfiles.append(buildfile)
                script = global_unintent(script)
                script = '\n'.join(script)
                spec = importlib.util.spec_from_loader(
                    'rosetta.build', loader=None, origin=buildfile)
                module = importlib.util.module_from_spec(spec)
                globals = module.__dict__
                scriptdir = buildfile.parent
                relbuildfile = buildfile.relative_to(scriptdir)

                def add_benchmark(*args, sources=None, basename=None, ppm=None, params=None):
                    for a in args:
                        if a in {'serial', 'cuda', 'omp_parallel', 'omp_task', 'omp_target', 'sycl'}:
                            ppm = a
                        elif isinstance(a, registry.GenParam) or isinstance(a, registry.SizeParam) or isinstance(a, registry.TuneParam):
                            params = (params or []) + [a]
                        else:
                            die(f"Unknown argument to add_benchmark in {buildfile}: {a}")

                    if sources is not None:
                        mysources = []
                        for s in sources:
                            mysources.append(scriptdir / mkpath(s))
                    else:
                        # Guess sources (same file)
                        mysources = [buildfile]
                    firstsource = mkpath(mysources[0])

                    # Generate a target name
                    rel = firstsource.parent.parent.relative_to(benchdir)

                    # Guess basename
                    if not basename:
                        basename = firstsource.stem
                        basename = removesuffix(basename, '.omp_parallel')
                        basename = removesuffix(basename, '.omp_task')
                        basename = removesuffix(basename, '.omp_target')
                        basename = removesuffix(basename, '.sycl')
                        basename = '.'.join(list(rel.parts) + [basename])

                    target = basename + '.' + ppm
                    pbsize = config.getint(basename, 'n')

                    bench = runner.Benchmark(basename=basename, target=target, exepath=None, ppm=ppm,
                                             configname=configname, buildtype=None, sources=mysources, pbsize=pbsize, params=params)
                    benchs.append(bench)

                globals['add_benchmark'] = add_benchmark
                globals['__file__'] = relbuildfile

                globals['GenParam'] = registry. GenParam
                globals['SizeParam'] = registry. SizeParam
                globals['TuneParam'] = registry.  TuneParam
                globals['runtime'] = registry. runtime
                globals['compiletime'] = registry. compiletime

                # Common PPMs for convenience
                globals['serial'] = 'serial'
                globals['cuda'] = 'cuda'
                globals['omp_parallel'] = 'omp_parallel'
                globals['omp_task'] = 'omp_task'
                globals['omp_target'] = 'omp_target'
                globals['sycl'] = 'sycl'
                globals['mpi'] = 'mpi'

                exec(script, module.__dict__)

    out = StringIO()
    print("# autogenerated by Rosetta gen-stage1.py\n", file=out)

    if configdepfiles:
        print("if (NOT ROSETTA_MAINTAINER_MODE)", file=out)

        print("# Build instructions were found in these", file=out)
        print("  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS", file=out)
        print('"' + ';'.join(pyescape(s)
              for s in configdepfiles) + '")', file=out)

        print("# These were searched for build instructions", file=out)
        print("  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS", file=out)
        print('"' + ';'.join(pyescape(s)
              for s in potentialbuildfiles if s not in configdepfiles) + '")', file=out)

        print("endif ()", file=out)
        print(file=out)

    for bench in benchs:
        if bench.ppm == 'serial':
            print(f"add_benchmark_serial({bench.basename}", file=out)
        elif bench.ppm == 'omp_parallel':
            print(f"add_benchmark_openmp_parallel({bench.basename}", file=out)
        elif bench.ppm == 'omp_task':
            print(f"add_benchmark_openmp_task({bench.basename}", file=out)
        elif bench.ppm == 'omp_target':
            print(f"add_benchmark_openmp_target({bench.basename}", file=out)
        elif bench.ppm == 'cuda':
            print(f"add_benchmark_cuda({bench.basename}", file=out)
        elif bench.ppm == 'sycl':
            print(f"add_benchmark_sycl({bench.basename}", file=out)
        else:
            die("Unhandled ppm")
        print("    PBSIZE", bench. pbsize, file=out)
        print("    SOURCES", *(cmakequote(bs.as_posix()) for bs in bench.sources), file=out)
        print("  )", file=out)

    updatefile(outfile, out.getvalue())


def main():
    #print("stage1 argv", sys.argv)
    parser = argparse.ArgumentParser(
        description="Generate CMakeLists.txt include file", allow_abbrev=False)
    parser.add_argument('--output', type=pathlib.Path)
    parser.add_argument('--problemsizefile', type=pathlib.Path)
    parser.add_argument('--benchdir', type=pathlib.Path)
    parser.add_argument('--builddir', type=pathlib.Path)
    parser.add_argument('--configname')
    # TODO: More extensive filter mechanisms
    parser.add_argument('--filter-include', '--filter', action='append',
                        help="Only look into filenames that contain this substring")
    args = parser.parse_args()

    gen_benchtargets(outfile=args.output, problemsizefile=args.problemsizefile, benchdir=args.benchdir,
                     builddir=args.builddir, configname=args.configname, filter=args.filter_include)


if __name__ == '__main__':
    retcode = main()
    if retcode:
        exit(retcode)
