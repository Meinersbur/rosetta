#ifndef ROSETTA_H_
#define ROSETTA_H_

#include <cstdlib>
#include <cstdio>
#include <string>
#include <chrono>
#include <cmath>
#include <variant>

// TODO: ROSETTA_PLATFORM_NVIDIA
// TODO: Get out of header
#if ROSETTA_PPM_CUDA
#include <cuda_runtime_api.h>
#endif

// From Google benchmark
// TODO: remove, make standalone
#include "benchmark.h"
#include "internal_macros.h"



#define BENCH_CUDA_TRY(call)                                                         \
  do {                                                                               \
    auto const status = (call);                                                      \
    if (cudaSuccess != status) { printf("CUDA call '" #call "' returned %d\n", status);  abort(); } \
  } while (0);


using benchmark::ClobberMemory;

class Scope;
class Iteration;
class AutoIteration;
template <typename I>
class Iterator;
class Range;
class State;



enum Measure {
     WallTime,
     UserTime,
     KernelTime, // TODO:
     OpenMPWTime,
     AccelTime,  // CUDA Event
     Cupti, // CUPTI duration from first to last event
     CuptiCompute, // CUPTI duration from first kernel launch to last kernel finish
     CuptiTransferToDevice, // CUPTI duration from start of first transfer to device start to last to finish
     CuptiTransferToHost, // CUPTI duration from start of first transfer from device start to last to finish
     MeasureLast = CuptiTransferToHost
};
constexpr int MeasureCount = MeasureLast+1;


// TODO: filter out duplicates
using duration_t = std::variant<std::monostate,
    std::chrono::duration<double,std::chrono::seconds::period>,  // lowest common denominator
    std::chrono::high_resolution_clock::duration, // for wall time
    std::chrono::duration<double,std::chrono::milliseconds::period>, // Used by CUDA events
    std::chrono::duration<uint64_t,std::chrono::nanoseconds::period> // Used by cupti
   >;


class IterationMeasurement {
    friend class Iteration;
    template <typename I>
    friend class Iterator;
    friend class State;
    friend class Rosetta;
    friend class Scope;
    friend class BenchmarkRun;
public:

private:
    // TODO: Make extendable (register user measures in addition to predefined ones)
    duration_t values[MeasureCount] ;
};



class Iteration {
  template <typename I>
  friend class Iterator;
   friend class State;
      friend class Rosetta;
      friend class Scope;
public :
  ~Iteration () {   // TODO: stop if not yet stopped
  }
   Scope scope() ;

void start();
void stop();

protected:
  explicit Iteration(State &state) : state(state) {}


  State &state;
};


class Scope {
    friend class Iteration;
    friend class AutoIteration;
    template <typename I>
    friend class Iterator;
    friend class Range;
    friend class State;
public:
    ~Scope() {
        it.stop();
    }

private:
    Scope(Iteration& it) : it(it) {
        it.start();
    }

    Iteration &it;
};

inline
Scope Iteration::scope() {
    return Scope(*this);
}




class AutoIteration : public Iteration {
    template <typename I>
    friend class Iterator;
public :
  ~AutoIteration () {}

private:
  AutoIteration(State &state) : Iteration(state), scope(*this) {
//printf("AutoIteration\n");
  }

  Scope scope;
};






class Range {
friend class State;
  public:
    Iterator<Iteration> begin() ;
     Iterator<Iteration> end()  ;

    private:
explicit Range(State &state) : state(state) {}

    State &state;
};

class BenchmarkRun;

class State {
  template <typename I>
      friend class Iterator;
   friend class Iteration;
   friend class Rosetta;
   friend class BenchmarkRun;
public:
     Iterator<AutoIteration>  begin() ;
     Iterator<AutoIteration>  end()   ;

Range manual() { return Range(*this) ; }

private:
    State (BenchmarkRun *impl) : impl(impl) {}
 // State (std::chrono::steady_clock::time_point startTime) : startTime(startTime) {}

  void start();
  void stop();
  int refresh();

  BenchmarkRun *impl;
};




template <typename I>
class Iterator {
    friend class Range;
    friend class State;
    friend class Rosetta;

public :
    typedef std::forward_iterator_tag iterator_category;
    typedef  I value_type;
    typedef  I &reference;
    typedef  I *pointer;
    typedef std::ptrdiff_t difference_type;


    BENCHMARK_ALWAYS_INLINE
        I operator*() const {
         //  printf("operator*()\n");  
        return I(state);
    }

    BENCHMARK_ALWAYS_INLINE
        Iterator& operator++() {
        assert(remaining > 0 );
        remaining -= 1;
        return *this;
    }

    BENCHMARK_ALWAYS_INLINE 
        bool operator!=(Iterator const& that) const {
        if (BENCHMARK_BUILTIN_EXPECT(remaining != 0, true)) return true;
        remaining = state.refresh();
        assert(remaining >= 0 );
        return remaining != 0;
    }

private:
    explicit Iterator(State &state, bool IsEnd) : state(state), isEnd(IsEnd) {}

    State &state;
    mutable int remaining = 0;
    bool isEnd;
};


inline Iterator<Iteration> Range::begin() { return Iterator<Iteration>(state, false); }
inline Iterator<Iteration> Range::end()   { return Iterator<Iteration>(state, true);  }


inline Iterator<AutoIteration>  State::begin() { return Iterator<AutoIteration>(*this, false); }
inline Iterator<AutoIteration>State::end()     { return Iterator<AutoIteration>(*this, true);}



#if 0
class CudaState {
public:
struct StateIterator;

CudaState (benchmark::State &gstate) : gstate(gstate) {}

  BENCHMARK_ALWAYS_INLINE StateIterator begin() { return StateIterator(*this, gstate.begin()); }
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

#if 0
  BENCHMARK_ALWAYS_INLINE
  int threads() const { return gstate.threads(); }

  BENCHMARK_ALWAYS_INLINE
  int thread_index() const { return gstate.thread_index(); }
#endif

  BENCHMARK_ALWAYS_INLINE
 benchmark::  IterationCount iterations() const {return gstate.iterations();}


  benchmark::UserCounters &getUserCounters () {return gstate.counters; }


struct StateIterator {
  typedef struct Value {
   explicit Value(CudaState &parent) : parent(parent) {
     #if ROSETTA_CUDA
     // printf("start\n");
     auto status = cudaEventCreate(&start);
     cudaEventRecord(start);
     #endif
    }
    ~Value() {
#if ROSETTA_CUDA
       //  printf("stop\n");
      cudaEvent_t stop;
       auto status = cudaEventCreate(&stop);
       cudaEventRecord(stop);
      // cudaThreadSynchronize();
      cudaEventSynchronize(stop);

       float msecs = 0;
       cudaEventElapsedTime(&msecs, start, stop);
       parent.getUserCounters()["GPU"] =  msecs; // FIXME: This is for the entire run, not a single iteration


         cudaEventDestroy(start);
        cudaEventDestroy(stop);
#endif
    }

  private:
#if ROSETTA_CUDA
  cudaEvent_t start;
#endif
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
#if ROSETTA_CUDA
    cudaEvent_t start;
#endif
};



#if ROSETTA_CUDA
typedef  CudaState State;
#else
typedef  benchmark::State State;
#endif


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

#endif

#endif /* ROSETTA_H_ */
