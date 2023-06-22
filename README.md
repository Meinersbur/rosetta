Parallel Programming Languages Rosetta Stone
============================================

[![Self-check](https://github.com/Meinersbur/rosetta/actions/workflows/check.yml/badge.svg)](https://github.com/Meinersbur/rosetta/actions/workflows/check.yml)

Rosetta is a combinations of a benchmarking framework (think of Google Benchmark) and a benchmark collection of codes in implemented multiple programming languages, styles, idioms and libraries.

Benchmarking Framework
----------------------

Rosetta includes a framework for benchmarking parallel programs. It currently supports the following parallel programming models (PPMs):

 * Sequential
 * OpenMP fork/join (`#pragma omp parallel for`)
 * OpenMP task (`#pragma omp task`)
 * OpenMP target offloading (`#pragma omp target`)
 * CUDA

Support for the following PPMs could be added in the future:

 * std::thread/pthreads
 * OpenACC
 * HIP
 * OpenMP
 * SYCL
 * UPC


Rosetta handles the execution and measurements of the programs it runs. The following platforms are supported:

 * Host OS (Wall time, user/kernel time, Max RSS)
   * getrusage
   * Win32
 * CUDA Toolkit (Cupti)

Support for the following platforms could be added in the future:

 * OMPT
 * AMD ROCm
 * Intel OneAPI (Level zero)


### Why not Google Benchmark?

Google Benchmarks has been designed for microbenchmarking. That is, benchmarks that run for at most some microseconds (Default time unit is nanoseconds). This makes it less useful for benchmarks the usually take longer. Specifically, in order to see effects of  NUMA or exceeding L3 cache sizes, programs need to run for longer.

 * Google Benchmark only measures only one timing metric at at time: Wall clock (real) time or CPU time. https://github.com/google/benchmark/blob/main/docs/user_guide.md#cpu-timers

 * Additional time measurements (such as GPU time) have to be done manually. https://github.com/google/benchmark/blob/main/docs/user_guide.md#manual-timing

 * Google Benchmark does not determine the spread of the measurements (e.g. variance, confidence intervals, quantils, ...). It assumes that stability can be reached by iterating often enough. This is infeasable if single iteration takes seconds or longer.

 * Google Benchmark is optimized for low overhead between iterations. The number of iterations of is determined in advance, such as the only overhead between iterations checking whether that number has been reached yet. Because of this low overhead, the timer is not stopped to exclude the overhead. However, if it needs to be paused (https://github.com/google/benchmark/blob/main/docs/user_guide.md#controlling-timers), the overhead is comparativelty significant for microbenchmarks.

 * Google Benchmark calls the entire benchmark code, including tearup and teardown multiple times with increasing number of iterations until a criterion is reached (https://github.com/google/benchmark/blob/398a8ac2e8e0b852fa1568dc1c8ebdfc743a380a/src/benchmark_runner.cc#L235). Only the last run is reported. For longer running benchmarks, the repeated tearup and teardown time may be signficant and cannot affort discarding previous runs.

 * https://github.com/google/benchmark/blob/main/docs/user_guide.md#multithreaded-benchmarks

 * https://github.com/google/benchmark/blob/main/docs/user_guide.md#memory-usage

In summery, Google Benchmark minimizes overhead-per-iteration but requires many iterations while Rosetta tries to make every iteration count and affords more measurements for statistics.


Benchmark Collection
--------------------

Comparable Projects
-------------------

### Benchmark Frameworks

 * Google Benchmark
 * https://github.com/UO-OACISS/apex

### Language Syntax Comparisons

 * http://99-bottles-of-beer.net/
 * [Rosetta Code](https://rosettacode.org/wiki/Rosetta_Code)

### Language Performance Comparisions

 * [The Computer Language Benchmarks Game](https://benchmarksgame-team.pages.debian.net/benchmarksgame/)
 * https://github.com/asad-iqbal-ali/OpenMP-vs-CUDA-Integration
 * https://github.com/mrchowmein/Cuda_Comparison

### Benchmark Collections

 * LLVM test-suite
 * SPEC OpenMP 2012

### Benchmark Collections in Multiple PPMS

 * NAS Parallel Benchmarks
 * [Rodinia](https://rodinia.cs.virginia.edu/doku.php)


### Run using Python
- Create a virtual environment and install necessary packages 
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
- Add Thrust in environment variable in `.zshrc`:
```shell
export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH
export Thrust_DIR=/usr/local/cuda-12.0/lib64/cmake/thrust
```
- Install `cmake` and `ninja-build`:
```shell
sudo apt install cmake
sudo apt-get install ninja-build
```
- Using Rosetta's own script: 
```
python /path/to/rosetta/rosetta.py <subcommand>
```
Subcommands are: `build`, `verify`, `bench` and a couple of others which can be listed with `-h`.
For example:
```shell
python rosetta.py -h 
python rosetta.py --build
python rosetta.py --bench
```

### Filters for programming models and programs

Rosetta benchmarks can be executed using two types of filters:
- Program filter:
  - `--filter-include-program-substr`: Include a specific program to the benchmark that contains this substring. 
  - `--filter-include-program-exact`: Include a specific program to the benchmark that exactly matches this string. 
  - `--filter-include-program-regex`: Include a specific program to the benchmark that matches this regex. 
  - `--filter-exclude-program-substr`: Exclude a specific program from the benchmark that contains this substring. 
  - `--filter-exclude-program-exact`: Exclude a specific program from the benchmark that exactly matches this string.
  - `--filter-exclude-program-regex`: Exclude a specific program from the benchmark that matches this regex. 
- Programming models filter: 
  - `--filter-include-ppm-substr`: Include a specific programming model to the benchmark that contains this substring. 
  - `--filter-include-ppm-exact`: Include a specific programming model to the benchmark that exactly matches this string. 
  - `--filter-include-ppm-regex`: Include a specific programming model to the benchmark that matches this regex. 
  - `--filter-exclude-ppm-substr`: Exclude a specific programming model from the benchmark that contains this substring.
  - `--filter-exclude-ppm-exact`: Exclude a specific programming model from the benchmark that exactly matches this string.
  - `--filter-exclude-ppm-regex`: Exclude a specific programming model from the benchmark that matches this regex. 

Example of using filters:
  - `python rosetta.py --bench --filter-include-program-substr assign --filter-include-program-regex ".*polybench.*" --filter-exclude-program-substr heat-3d --filter-include-ppm-substr serial`
  - `python rosetta.py --bench --filter-include-program-substr assign`
  - `python rosetta.py --bench --filter-include-program-substr assign --filter-include-ppm-substr serial --filter-include-ppm-substr cuda`
  - `python rosetta.py --bench --filter-exclude-program-substr polybench --filter-exclude-ppm-regex "openmp.*"`
  - `python rosetta.py --bench --filter-include-program-exact idioms.assign`
  - `python rosetta.py --bench --filter-include-program-regex ".*mm$" --filter-include-ppm-substr serial`
  - `python rosetta.py --bench --filter-exclude-program-regex ".*polybench.*" --filter-include-ppm-substr serial`
  - `python rosetta.py --bench --filter-exclude-ppm-substr cuda --filter-exclude-ppm-substr openmp-parallel --filter-exclude-program-substr polybench`

To use filtering please make sure `ROSETTA_BENCH_FILTER:STRING` is not set in `CMakeCache.txt` in `build/defaultbuild` directory:
```shell
//Benchmark filter switches
ROSETTA_BENCH_FILTER:STRING=
```

To run any of the unittests, execute the relevant test file. For run unittest for filtering:
```shell
python rosetta/test/test_filtering.py
```

### (Optional) Run with predefined problem sizes
- Optionally, you can utilize the `--problemsizefile` argument to specify a predefined problem sizes for the benchmarks. Available values for `--problemsizefile` argument are: `mini`, `small`, `medium`, `large`, and `extralarge`. If the `--problemsizefile` argument is not provided, the default `medium` problemsize file will be used.
```shell
python rosetta.py --bench --problemsizefile large
```
- You can also verify the different problem size files excluding CUDA PPM:
```shell
python rosetta.py --verify --problemsizefile small --filter-exclude-ppm-exact cuda
```
