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




class dyn_array_base{
protected:
    dyn_array_base(BenchmarkRun* impl, int size, bool verify) ;
    dyn_array_base(dyn_array_base&& that)  : impl(that.impl), size(that.size), verify(that.verify)   { that.impl=nullptr; that.verify= false;  that.size=0; }
    ~dyn_array_base() ;

    BenchmarkRun *impl;
    size_t size;
    bool verify ;
};

template<typename T> 
class dyn_array : dyn_array_base {
    friend class State;
public:
    ~dyn_array() { 
        if (verify) verifydata();
        verify=false;
        delete[] mydata;  
        mydata = nullptr;
    }

    dyn_array(const dyn_array &that) = delete;
    dyn_array & operator=(const dyn_array &that) =delete;


    dyn_array(dyn_array&& that) : dyn_array_base(std::move(that)), mydata(that.mydata) {
        that.mydata=nullptr;
    }


    T* data() { return mydata ;};
    const    T* data()const  { return mydata ;};

    void zerodata() {
        memset(data, '\0', size);
    }

    void fakedata() {  DataHandler<T>(impl).fake(mydata,size/sizeof(T)); }
    void verifydata() {
#if ROSETTA_VERIFY
        DataHandler<T>(impl).verify(mydata,size/sizeof(T));
#else
        // Don't do anything in benchmark mode
#endif 
    }




    // TODO: realloc
private:
    dyn_array(BenchmarkRun* impl, int count, bool verify  ) : dyn_array_base(impl, count * sizeof(T), verify ), mydata(new T[count]) {      }

    // typed, to ensure TBAA can be effective
        T* mydata;
};




 



class State {
  template <typename I>
      friend class Iterator;
   friend class Iteration;
   friend class Rosetta;
   friend class BenchmarkRun;
   friend class dyn_array_base;
   template <typename T>
   friend class dyn_array;
public:
     Iterator<AutoIteration>  begin();
     Iterator<AutoIteration>  end();

Range manual() { return Range(*this) ; }

template<typename T>
struct InternalData {
    size_t size;
    T Data[];
};

// TODO: return some smart ptr, we are C++ after all
// TODO: ensure alignment
template<typename T>
    T* malloc(size_t count) {   
  auto result = (InternalData<T>*) ::malloc(sizeof(InternalData<T>) + sizeof( T) * count );
  result->size = count * sizeof(T);
  addAllocatedBytes(count * sizeof(T));
  return &result->Data[0];
}

    template<typename T>
    dyn_array<T> alloc_array(size_t count, bool verify= false) {   
        return  dyn_array<T>( impl, count ,verify );
    }

    template<typename T>
    dyn_array<T> calloc_array(size_t count , bool verify=false) {   
        auto  result =  dyn_array<T>( impl, count, verify  );
        result.zerodata();
        return result;
    }

    template<typename T>
    dyn_array<T> fakedata_array(size_t count, bool verify=false) {   
        auto  result =  dyn_array<T>( impl, count ,verify );
        result.fakedata();
        return result;
    }


template<typename T>
void fakedata(T *data, size_t count) {
    DataHandler<T>(impl).fake(data,count);
}


template<typename T>
void verifydata(T* data, size_t count) {
#if ROSETTA_VERIFY
    DataHandler<T>(impl).verify(data,count);
#else
    // Don't do anything in benchmark mode
#endif 
}

template<typename T>
void free (T *ptr) { 
    auto internaldata =(InternalData<T>*)  (((char*)ptr) -offsetof(InternalData<T>, Data)); 

   subAllocatedBytes(internaldata->size);
   ::free(internaldata);
}

private:
    State (BenchmarkRun *impl) : impl(impl) {}
 // State (std::chrono::steady_clock::time_point startTime) : startTime(startTime) {}

  void start();
  void stop();
  int refresh();

  void addAllocatedBytes(size_t size);
  void subAllocatedBytes(size_t size);

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
