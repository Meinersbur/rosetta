#ifndef ROSETTA_H_
#define ROSETTA_H_

#include <cstdlib>
#include <cstdio>
#include <string>
#include <benchmark/benchmark.h>


class CudaState {

public:
struct StateIterator;

CudaState (benchmark::State &gstate) : gstate(gstate) {}

  BENCHMARK_ALWAYS_INLINE StateIterator begin() { return StateIterator(*this,  gstate.begin()); }
  BENCHMARK_ALWAYS_INLINE StateIterator end()   { return StateIterator(*this, gstate.end());};

    bool KeepRunning() { return  gstate.KeepRunning(); }

     bool KeepRunningBatch(benchmark:: IterationCount n) { return gstate.KeepRunningBatch(n); } 

  void PauseTiming() { return gstate.PauseTiming(); }

    void ResumeTiming() { return gstate.ResumeTiming(); }

      void SkipWithError(const char* msg) {return gstate.SkipWithError(msg); }

        bool error_occurred() const { return gstate.error_occurred(); }

          void SetIterationTime(double seconds) {return gstate.SetIterationTime(seconds);  }

  void SetBytesProcessed(int64_t bytes) { return gstate.SetBytesProcessed(bytes) ;}

  int64_t bytes_processed()const  { return gstate.bytes_processed(); }

BENCHMARK_ALWAYS_INLINE void SetComplexityN(int64_t complexity_n)  {return gstate.SetComplexityN(complexity_n); }

  BENCHMARK_ALWAYS_INLINE
  int64_t complexity_length_n() const {return complexity_length_n(); }

    BENCHMARK_ALWAYS_INLINE
  void SetItemsProcessed(int64_t items) { return gstate.SetItemsProcessed(items); }

BENCHMARK_ALWAYS_INLINE
  int64_t items_processed() const { return  gstate.items_processed(); }

  void SetLabel(const char* label) {return gstate.SetLabel(label);}

    BENCHMARK_ALWAYS_INLINE
  int64_t range(std::size_t pos = 0) const {return gstate.range(pos);}

  BENCHMARK_ALWAYS_INLINE
 benchmark::  IterationCount iterations() const {return gstate.iterations();}


  benchmark::UserCounters &getUsetCounters () {return gstate.counters; }
  int getThreadIndex() const  { return gstate.thread_index; }
int getNumThreads() const  { return gstate.threads; }

struct StateIterator {
  typedef struct Value {
   explicit Value(CudaState &parent) : parent(parent) {
     // printf("start\n");   
     auto status = cudaEventCreate(&start);   
     cudaEventRecord(start);
    }
    ~Value() {
       //  printf("stop\n");           
      cudaEvent_t stop;
       auto status = cudaEventCreate(&stop);
       cudaEventRecord(stop);
      // cudaThreadSynchronize();
      cudaEventSynchronize(stop);

       float msecs = 0;
       cudaEventElapsedTime(&msecs, start, stop);
       parent.getUsetCounters()["GPU"] =  msecs; // FIXME: This is for the entire run, not a single iteration


         cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

  private:
  cudaEvent_t start;
    CudaState &parent;
  } Value;
  typedef std::forward_iterator_tag iterator_category;
  typedef  Value value_type;
  typedef  Value reference;
  typedef  Value pointer;
  typedef std::ptrdiff_t difference_type;

  friend class CudaState;

  BENCHMARK_ALWAYS_INLINE
  StateIterator(CudaState &parent, benchmark::State::StateIterator git): parent(parent), git(std::move(git)) {}

 public:
  BENCHMARK_ALWAYS_INLINE
  Value operator*() const {  git.operator*();
    return Value(parent);
   }

  BENCHMARK_ALWAYS_INLINE
  StateIterator& operator++() {
    git.operator++();
    return *this;
  }

  BENCHMARK_ALWAYS_INLINE
  bool operator!=(StateIterator const& that) const {
    return git.operator!=(that.git);
  }

 private:
  benchmark::State::StateIterator git;
  CudaState &parent;
};


  private:
    benchmark::State &gstate;
    cudaEvent_t start;
};




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
