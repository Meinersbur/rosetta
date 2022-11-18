#! /bin/sh
SCRIPTDIR=$(dirname $0)
env "PYTHONPATH=${SCRIPTDIR}/rosetta/src" python3 -m rosetta.benchmark "--rootdir=${SCRIPTDIR}" "$@"
