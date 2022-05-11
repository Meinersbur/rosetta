#ifndef ROSETTA_H_
#define ROSETTA_H_

#include <cstdlib>
#include <cstdio>
#include <string>
#include <benchmark/benchmark.h>

#ifndef DEFAULT_N
  #define DEFAULT_N 128
#endif

#define BENCH_CUDA_TRY(call)                                                         \
  do {                                                                               \
    auto const status = (call);                                                      \
    if (cudaSuccess != status) {  \
    fprintf(stdout, "Call return error code %d: %s\n", status, #call); \
        abort(); \
     } \
  } while (0);

typedef void BenchmarkFuncTy(benchmark::State& , int);

class RosettaBenchmark {
private:
const char *name;
  BenchmarkFuncTy *func;
  RosettaBenchmark *next;
public:
    RosettaBenchmark(const char *name, BenchmarkFuncTy &func);

    const char *getName() const {return name;}
    BenchmarkFuncTy  *getFunc() const {return func;}
    RosettaBenchmark*getNext() const {return next;}
};

#define ROSETTA_BENCHMARK(NAME) \
  static RosettaBenchmark StaticInitializer{#NAME, NAME};

#endif /* ROSETTA_H_ */
