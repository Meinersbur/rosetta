# -*- coding: utf-8 -*-

import math

from .util.cmdtool import *
from .util.support import *
from .util import invoke
from .common import *
from .util.cmdtool import *
from .util.support import *
from .common import *
from .runner import Benchmark,get_problemsizefile,get_problemsize
from . import runner, registry




def get_refpath(bench, refdir, problemsizefile):
    pbsize = get_problemsize(bench, problemsizefile=problemsizefile)
    reffilename = f"{bench.name}.{pbsize}.reference_output"
    refpath = refdir / reffilename
    return refpath



def ensure_reffile(bench: Benchmark, refdir, problemsizefile):
    refpath = get_refpath(bench, refdir=refdir,  problemsizefile=problemsizefile)




    # Get the reference implementation
    # TODO : Use the one from reference builddir if any
    refbench = bench.comparable .reference


    if refpath.exists():
        # Reference output already exists; check that it is the latest
        benchstat = refbench.exepath.stat()
        refstat = refpath.stat()
        if benchstat.st_mtime < refstat.st_mtime:
            print(f"Reference output of {refbench.exepath} ({benchstat.st_mtime}) already exists at {refpath} ({refstat.st_mtime}) an is up-to-date")
            return
        print(f"Reference output {refpath} an is out-of-date")
        refpath.unlink()


    # Invoke reference executable and write to file
    args = [refbench.exepath, f'--verify', f'--verifyfile={refpath}']
    if problemsizefile:
        args.append(f'--problemsizefile={problemsizefile}')
    invoke.call(*args, print_command=True)
    if not refpath.is_file():
        print(f"{refpath} not been written?")
        assert refpath.is_file()

    print(f"Reference output of {bench.name} written to {refpath}")



def ensure_reffiles(refdir, problemsizefile, filterfunc=None, srcdir=None):
    problemsizefile = get_problemsizefile(
        srcdir=srcdir, problemsizefile=problemsizefile)
    for bench in runner. benchmarks:
        if filterfunc and not filterfunc(bench):
            continue
        ensure_reffile(bench, refdir=refdir, problemsizefile=problemsizefile)


def run_verify(problemsizefile, filterfunc=None, srcdir=None, refdir=None):
    problemsizefile = get_problemsizefile(
        srcdir=srcdir, problemsizefile=problemsizefile)



    refdir.mkdir(exist_ok=True, parents=True)

    for e in registry.benchmarks:
        if filterfunc and not filterfunc(e):
            continue

        ensure_reffile(e, refdir=refdir, problemsizefile=problemsizefile)

        exepath = e.exepath
        refpath = get_refpath(
            e, refdir=refdir, problemsizefile=problemsizefile)
        pbsize = get_problemsize(e, problemsizefile=problemsizefile)

        testoutpath = request_tempfilename(
            subdir='verify', prefix=f'{e.name}_{e.ppm}_{pbsize}', suffix='.testout')


        args = [exepath, f'--verify', f'--verifyfile={testoutpath}']
        if problemsizefile:
            args.append(f'--problemsizefile={problemsizefile}')
        invoke.call(*args, return_stdout=True, print_command=True)

        with refpath.open() as fref, testoutpath.open() as ftest:
            while True:
                refline = fref.readline()
                testline = ftest.readline()

                # Reached end-of-file?
                if not refline and not testline:
                    break

                refspec, refdata = refline.split(':', maxsplit=1)
                refspec = refspec.split()
                refkind = refspec[0]
                refformat = refspec[1]
                refdim = int(refspec[2])
                refshape = [int(i) for i in refspec[3:3 + refdim]]
                refname = refspec[3 +
                                  refdim] if len(refspec) > 3 + refdim else None
                refcount = prod(refshape)

                refdata = [float(v) for v in refdata.split()]
                if refcount != len(refdata):
                    die(
                        f"Unexpected array items in {refname}: {refcount} vs {len(refdata)}")

                testspec, testdata = testline.split(':', maxsplit=1)
                testspec = testspec.split()
                testkind = testspec[0]
                testformat = testspec[1]
                testdim = int(testspec[2])
                testshape = [int(i) for i in testspec[3:3 + testdim]]
                testname = testspec[3 +
                                    testdim] if len(testspec) > 3 + testdim else None
                testcount = prod(testshape)

                testdata = [float(v) for v in testdata.split()]
                if testcount != len(testdata):
                    die(
                        f"Unexpected array items in {testname}: {testcount} vs {len(testdata)}")

                if refname is not None and testname is not None and refname != testname:
                    die(f"Array names {refname} and {testname} disagree")

                errsfound  = 0
                for i, (refv, testv) in enumerate(zip(refdata, testdata)):
                    coord = [str((i // prod(refshape[0:j])) % refshape[j])
                             for j in range(0, refdim)]
                    coord = '[' + ']['.join(coord) + ']'

                    if math.isnan(refv) and math.isnan(testv):
                        print(
                            f"WARNING: NaN in both outputs at {refname}{coord}")
                        continue
                    if math.isnan(refv):
                        die(
                            f"Array data mismatch: Ref contains NaN at {refname}{coord}")
                    if math.isnan(testv):
                        die(
                            f"Array data mismatch: Output contains NaN at {testname}{coord}")

                    mid = (abs(refv) + abs(testv)) / 2
                    absd = abs(refv - testv)
                    if mid == 0:
                        reld = 0 if absd == 0 else math.inf
                    else:
                        reld = absd / mid
                    #if refv != absd :
                    if reld > 1e-4:  # TODO: Don't hardcode difference
                        if errsfound == 0:
                            print(f"While comparing {refpath} and {testoutpath}:")
                        print( f"Array data mismatch: {refname}{coord} = {refv} != {testv} = {testname}{coord} (Delta: {absd}  Relative: {reld})")
                        errsfound += 1

                    if errsfound >= 20:
                        die (f"Found at least {errsfound} differences; output considered incorrect")

        if errsfound:
            die(f"Found {errsfound} output differences; output considered incorrect")

        print(f"Output of {e.exepath} considered correct")
