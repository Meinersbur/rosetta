Parallel Programming Languages Rosetta Stone
============================================

Rosetta is a combinations of a benchmarking framework (think of Google Benchmark) and a benchmark collection of codes in implemented multiple programming languages, styles, idioms and libraries.

Benchmarking Framework
----------------------

Rosatta includes a framework for benchmarking parallel programs. It currently supports the following parallel programming models (PPMs):

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


Rosatta handles the execution and measurements of the programs it runs. The following platforms are supported:

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
