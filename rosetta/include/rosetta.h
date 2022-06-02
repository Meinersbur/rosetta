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

// For ssize_t
#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif


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

template <typename T>
class DataHandler;

class DataHandlerBase {
    friend class State;
protected:
    explicit DataHandlerBase(BenchmarkRun* impl) : impl(impl) {}

    BenchmarkRun *impl;
};

template <>
class DataHandler<double> :  public DataHandlerBase {
    friend class State;
public:
    explicit DataHandler(BenchmarkRun* impl) : DataHandlerBase(impl) {}

    void fake(double *data, ssize_t count);
    void verify(double *data, ssize_t count);
};


class State {
  template <typename I>
      friend class Iterator;
   friend class Iteration;
   friend class Rosetta;
   friend class BenchmarkRun;
public:
     Iterator<AutoIteration>  begin();
     Iterator<AutoIteration>  end();

Range manual() { return Range(*this) ; }

// TODO: return some smart ptr, we are C++ after all
template<typename T>
T* malloc(size_t count) {
  auto result = new T[count]; // TODO: alignment
  addAllocatedBytes(count * sizeof(T));
  return result;
}

template<typename T>
void fakedata(T *data, size_t count) {
    DataHandler<T>(impl).fake(data,count);

}

template<typename T>
void verifydata(T* data, size_t count) {
    DataHandler<T>(impl).verify(data,count);
}

template<typename T>
void free (T *ptr) {
    // TODO: subtract allocated bytes 
    delete [] ptr;
}

private:
    State (BenchmarkRun *impl) : impl(impl) {}
 // State (std::chrono::steady_clock::time_point startTime) : startTime(startTime) {}

  void start();
  void stop();
  int refresh();

  void addAllocatedBytes(size_t allocatedSize);

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



#endif /* ROSETTA_H_ */
