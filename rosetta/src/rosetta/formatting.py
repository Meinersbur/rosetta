# -*- coding: utf-8 -*-

"""
Formatting of Rosetta's own source files using clang-format, cmake-format, and autopep8
"""

from .common import *
import pathlib
from .util import invoke


clangformat_options = ['--color=1']
cmakeformat_options = [
    '--line-width',
    '120',
    '--separate-ctrl-name-with-space',
    'true',
    '--enable-markup',
    'false',
    '--output-encoding',
    'utf-8',
    '--line-ending',
    'unix',
]
autopep8_options = ['--max-line-length', '120', '-aa']
black_options = ['--line-length', '120', '--skip-string-normalization', '--color']


def update_format(srcdir: pathlib.Path):
    files = updateable_files(srcdir=srcdir, framework=True, benchmarks=True)

    invoke.run(
        'clang-format', '-i', *(clangformat_options + files.cpp), print_command=True, onerror=invoke.Invoke.IGNORE
    )

    invoke.run('cmake-format', '--in-place', *(cmakeformat_options + files.cmake), print_command=True)

    invoke.run('black', *(black_options + files.py), print_command=True)


def check_format(srcdir: pathlib.Path):
    files = updateable_files(srcdir=srcdir, framework=True, benchmarks=True)
    anyerror = False

    retcode = invoke.run(
        'clang-format',
        '-Werror',
        '--dry-run',
        *(clangformat_options + files.cpp),
        print_command=True,
        onerror=invoke.Invoke.IGNORE
    )
    if retcode:
        anyerror = True

    retcode = invoke.run(
        'cmake-format',
        '--check',
        *(cmakeformat_options + files.cmake),
        print_command=True,
        onerror=invoke.Invoke.IGNORE
    )
    if retcode:
        anyerror = True

    retcode = invoke.run(
        'black', '--check', '--diff', *(black_options + files.py), print_command=True, onerror=invoke.Invoke.IGNORE
    )
    if retcode:
        anyerror = True

    if anyerror:
        print("")
        print("Re-formatting necessary (or formatter executable not found)")
        print("Run rosetta.py --update-format to fix")
        exit(1)


class UpdateteableFiles:
    def __init__(self):
        self.cpp = []
        self.cmake = []
        self.py = []


def updateable_files(srcdir: pathlib.Path, framework: bool = False, benchmarks: bool = False):
    result = UpdateteableFiles()
    if framework:
        rosettadir = srcdir / 'rosetta'
        result.cpp += rosettadir.rglob('*.cpp')
        result.cpp += rosettadir.rglob('*.h')
        result.cpp += rosettadir.rglob('*.hpp')
        result.cmake.append(srcdir / 'CMakeLists.txt')
        result.cmake += rosettadir.rglob('*.cmake')
        result.cmake += rosettadir.rglob('CMakeLists.txt')
        result.py += rosettadir.rglob('*.py')
        result.py.append(srcdir / 'rosetta.py')

    if benchmarks:
        benchmarkdir = srcdir / 'benchmarks'
        result.cpp += benchmarkdir.rglob('*.h')
        result.cpp += benchmarkdir.rglob('*.cu')

        # clang-format does a terrible job formatting pragmas, so we currently don't autoformat them
        # https://github.com/llvm/llvm-project/issues/39271
        result.cpp += (
            f
            for f in benchmarkdir.rglob('*.cxx')
            if not f.name.endswith('omp_parallel.cxx')
            and not f.name.endswith('omp_task.cxx')
            and not f.name.endswith('omp_target.cxx')
        )

    return result
