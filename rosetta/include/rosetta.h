#ifndef ROSETTA_H_
#define ROSETTA_H_

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>
#include <variant>
#include <iomanip>


// TODO: ROSETTA_PLATFORM_NVIDIA
// TODO: Get out of header
#if ROSETTA_PPM_CUDA
#include <cuda_runtime_api.h>
#endif
#if ROSETTA_PPM_CUDA || ROSETTA_PLATFORM_NVIDIA
#include "rosetta-cuda.h"
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




using benchmark::ClobberMemory;

class Scope;
class Iteration;
class AutoIteration;
template <typename I>
class Iterator;
class Range;
class State;


typedef int idx_t;    // ssize_t / ptrdiff_t
typedef int pbsize_t; // ssize_t ?

#ifdef ROSETTA_REALTYPE
typedef ROSETTA_REALTYPE real_t;
#else
typedef double real_t;
#endif
typedef real_t real;



template <typename T, int DIMS>
class multarray;


template <typename T, int... __L>
class array;

template <int64_t>
class _inttype {
public:
  typedef int64_t type;
};

template <int64_t...>
class _dimlengths; // Never instantiate, used to store a sequence of integers, namely the size of several dimensions


template <size_t DIMS, int64_t... DIMLENGTHS>
struct _make_dimlengths {
  using type = /*recursion call*/ typename _make_dimlengths<DIMS - 1, 0, DIMLENGTHS...>::type;
};


template <int64_t... DIMLENGTHS>
struct _make_dimlengths<0, DIMLENGTHS...> {
  using type = _dimlengths<DIMLENGTHS...>;
};




template <typename DIMLENGTHS>
struct _make_tuple;

template <int64_t... DIMLENGTHS>
struct _make_tuple<_dimlengths<DIMLENGTHS...>> {
  using type = typename std::tuple<typename _inttype<DIMLENGTHS>::type...>;
};



template <int64_t First, int64_t... Rest>
class _unqueue {
public:
  typedef _dimlengths<Rest...> rest;
  typedef _dimlengths<First> first;
  static const int64_t value = First;
};

template <typename DIMLENGTH>
class _unqueue_dimlengths;

template <int64_t First, int64_t... Rest>
class _unqueue_dimlengths<_dimlengths<First, Rest...>> {
public:
  typedef _dimlengths<Rest...> rest;
  typedef _dimlengths<First> first;
  static const int64_t value = First;
};



template <typename SEQ>
struct _tuple_rest_extractor;

template <size_t... SEQ>
struct _tuple_rest_extractor<std::index_sequence<SEQ...>> {
  template <typename TUPLE>
  static auto get_tuple(TUPLE tuple) { return std::make_tuple(std::get<SEQ + 1>(tuple)...); };
};



template <typename TUPLE>
class _unqueue_tuple;

template <typename First, typename... Rest>
class _unqueue_tuple<std::tuple<First, Rest...>> {

public:
  using TupleTy = std::tuple<First, Rest...>;
  typedef std::tuple<Rest...> rest;
  typedef First first;

  static first get_first(TupleTy tuple) { return std::get<0>(tuple); }
  static rest get_rest(TupleTy tuple) { return _tuple_rest_extractor<std::make_index_sequence<sizeof...(Rest)>>::get_tuple(tuple); }
};



// from https://en.cppreference.com/w/cpp/utility/integer_sequence
template <typename Array, std::size_t... I>
auto a2t_impl(Array a, std::index_sequence<I...>) {
  return std::make_tuple(a[I]...);
}

template <typename T, std::size_t N, typename Indices = std::make_index_sequence<N>>
auto a2t(std::array<T, N> a) {
  return a2t_impl(a, Indices{});
}



template <typename T, std::size_t N>
auto a2v(std::array<T, N> a) {
  std::vector<T> v;
  v.reserve(N);
  for (auto val : a)
    v.push_back(val);
  return v;
}


template <typename T, std::size_t N, typename Is = typename std::make_index_sequence<N>>
struct t2v_helper;

template <typename T, std::size_t N>
struct t2v_helper<T, N, std::index_sequence<>> {
  static void add_elements(std::vector<T> &v, typename _make_tuple<typename _make_dimlengths<N>::type>::type t) {}
};


template <typename T, std::size_t N, size_t I, size_t... IRest>
struct t2v_helper<T, N, std::index_sequence<I, IRest...>> {
  static void add_elements(std::vector<T> &v, typename _make_tuple<typename _make_dimlengths<N>::type>::type t) {
    v.push_back(std::get<I>(t));
    t2v_helper<T, N, std::index_sequence<IRest...>>::add_elements(v, t);
  }
};

template <typename T, std::size_t N>
auto t2v(typename _make_tuple<typename _make_dimlengths<N>::type>::type t) {
  std::vector<T> v;
  v.reserve(N);
  t2v_helper<T, N>::add_elements(v, t);
  return v;
}



template <typename TUPLE>
static ssize_t get_stride(TUPLE tuple) {
  static_assert(std::tuple_size<TUPLE>::value >= 1);
  if constexpr (std::tuple_size<TUPLE>::value == 1) {
    return std::get<0>(tuple);
  }
  if constexpr (std::tuple_size<TUPLE>::value > 1) {
    return _unqueue_tuple<TUPLE>::get_first(tuple) * get_stride(_unqueue_tuple<TUPLE>::get_rest(tuple));
  }
  return 1;
}



template <typename DIMLENGTHS>
class __dimlengths_inttype;

template <int64_t... DIMLENGTHS>
class __dimlengths_inttype<_dimlengths<DIMLENGTHS...>> {
public:
  typedef int64_t type;
};






template <typename T /*Elt type*/, int Togo /*coordinates to go*/>
class _multarray_partial_subscript {
  static_assert(Togo > 1, "Specialization for more-coordinates-to-come");
  static const int nTogo = Togo;

  typedef _multarray_partial_subscript<T, Togo - 1> subty;
  using RemainingLengthsTy = typename _make_tuple<typename _make_dimlengths<Togo>::type>::type;

  T *data;
  RemainingLengthsTy remainingLengths;

public:
  _multarray_partial_subscript(T *data, RemainingLengthsTy remainingLengths)
      : data(data), remainingLengths(remainingLengths) {}

public:
  subty operator[](int64_t i) /*TODO: const*/ {
    auto len = _unqueue_tuple<RemainingLengthsTy>::get_first(remainingLengths);
    assert(0 <= i);
    assert(i < len);

    auto rest = _unqueue_tuple<RemainingLengthsTy>::get_rest(remainingLengths);
    return _multarray_partial_subscript<T, nTogo - 1>(data + i * get_stride(rest), rest);
  }
}; // class _multarray_partial_subscript



template <typename T>
class _multarray_partial_subscript<T, 1> {
  static constexpr int nTogo = 1;

  using RemainingLengthsTy = size_t;

  T *data;
  RemainingLengthsTy remainingLength;

public:
  _multarray_partial_subscript(T *data, std::tuple<int64_t> remainingLength)
      : data(data), remainingLength(std::get<0>(remainingLength)) {}

public:
  T &operator[](int64_t i) {
    assert(0 <= i);
    assert(i < remainingLength);
    return data[i];
  }
}; // class _multarray_partial_subscript



static const char * indent(int amount) {
    static const char *whitespace =
"                                                                                "; // 80 spaces
 assert(amount >= 0);
    assert (amount <= 80);
    return &whitespace[80 - amount];
}

template<typename T>
void dumpArray(T *data,std::tuple<int64_t,int64_t> DimLengths, const char *d) {
    size_t dlen = d ? strlen(d) : 0; // TODO: d should be taken from the allocation call
    for (int i = 0; i < std::get<0>(DimLengths); i += 1) {
        if (i == 0) {
            if (d)
                std::cerr << d << " = [ ";
            else
                std::cerr << "[ ";
        }
        else { 
            if (d)
                std::cerr << indent(dlen) << "     ";
            else 
                std::cerr << "  ";
        }
        for (int j = 0; j < std::get<1>(DimLengths); j += 1) {
                if (j > 0) std::cerr << " ";
                std::cerr <<  std::setw(4)  << data[i *std::get<1>(DimLengths) + j ] ;
        }
        if (i!=std::get<0>(DimLengths)-1)
        std::cerr << "\n"; else 
        std::cerr << " ]" << std::endl;
    }
}


template <typename T, int DIMS>
class multarray {
  typedef T ElementType;
  static const auto Dims = DIMS;
  using __L = typename _make_dimlengths<DIMS>::type;
  using TupleTy = typename _make_tuple<__L>::type;
  using PtrTy = T *;

private:
  PtrTy data;
  TupleTy DimLengths;

public:
  multarray(T *data, TupleTy lengths) : data(data), DimLengths(lengths) {}

  template <typename Dummy = void>
  typename std::enable_if<std::is_same<Dummy, void>::value && (DIMS == 1), T &>::type
  operator[](int64_t i) {
    auto len = std::get<0>(DimLengths);
    assert(0 <= i);
    assert(i < len);
    return data[i];
  }

  template <typename Dummy = void>
  typename std::enable_if<std::is_same<Dummy, void>::value && (DIMS > 1), _multarray_partial_subscript<T, DIMS - 1>>::type
  operator[](int64_t i) {
    auto len = _unqueue_tuple<TupleTy>::get_first(DimLengths);
    assert(0 <= i);
    assert(i < len);

    auto rest = _unqueue_tuple<TupleTy>::get_rest(DimLengths);
    // std::cerr << "DIMS=" << DIMS << " i=" <<i << " len=" << len <<  " i*get_stride(rest)=" << i* get_stride(rest) << "\n";
    return _multarray_partial_subscript<T, DIMS - 1>(data + i * get_stride(rest), rest);
  }



  void dump() const {
    dumpArray(data, DimLengths, nullptr);
  }

  void dump(const char *d) const {
      dumpArray(data, DimLengths, d);
  }
}; // class multarray


// TODO: Make extensible depending on PPM.
enum Measure {
  WallTime,
  UserTime,
  KernelTime, 
  OpenMPWTime,
  AccelTime,             // CUDA Event
  Cupti,                 // CUPTI duration from first to last event
  CuptiCompute,          // CUPTI duration from first kernel launch to last kernel finish
  CuptiTransferToDevice, // CUPTI duration from start of first transfer to device start to last to finish (TODO: Would be nice to subtract the time when no transfer is active)
  CuptiTransferToHost,   // CUPTI duration from start of first transfer from device start to last to finish
  MeasureLast = CuptiTransferToHost
};
constexpr int MeasureCount = MeasureLast + 1;



class Iteration {
  template <typename I>
  friend class Iterator;
  friend class State;
  friend class Rosetta;
  friend class Scope;

public:
  ~Iteration() { // TODO: stop if not yet stopped
  }
  Scope scope();

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
  Scope(Iteration &it) : it(it) {
    it.start();
  }

  Iteration &it;
};

inline Scope Iteration::scope() {
  return Scope(*this);
}



class AutoIteration : public Iteration {
  template <typename I>
  friend class Iterator;

public:
  ~AutoIteration() {}

private:
  AutoIteration(State &state) : Iteration(state), scope(*this) {
    // printf("AutoIteration\n");
  }

  Scope scope;
};



class Range {
  friend class State;

public:
  Iterator<Iteration> begin();
  Iterator<Iteration> end();

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
  explicit DataHandlerBase(BenchmarkRun *impl) : impl(impl) {}

  BenchmarkRun *impl;
};

template <>
class DataHandler<double> : public DataHandlerBase {
  friend class State;

public:
  explicit DataHandler(BenchmarkRun *impl) : DataHandlerBase(impl) {}

  void fake(double *data, ssize_t count);
  void verify(double *data, ssize_t count, std::vector<size_t> dims, std::string_view name);
};



class dyn_array_base {
protected:
  dyn_array_base(BenchmarkRun *impl, int size, bool verify, std::vector<size_t> dims, std::string_view name);
  dyn_array_base(dyn_array_base &&that) : impl(that.impl), size(that.size), verify(that.verify), dims(std::move(that.dims)), name(std::move(that.name)) {
    that.impl = nullptr;
    that.verify = false;
    that.size = 0;
  }
  ~dyn_array_base();

  BenchmarkRun *impl;
  size_t size;
  bool verify;
  std::vector<size_t> dims;
  std::string name;
};



template <typename T>
class dyn_array : dyn_array_base {
  friend class State;
  template <typename X, size_t DIMS>
  friend class owning_array;

public:
  ~dyn_array() {
    if (verify)
      verifydata();
    verify = false;
    delete[] mydata;
    mydata = nullptr;
  }

  dyn_array(const dyn_array &that) = delete;
  dyn_array &operator=(const dyn_array &that) = delete;


  dyn_array(dyn_array &&that) : dyn_array_base(std::move(that)), mydata(that.mydata) {
    that.mydata = nullptr;
  }


  T *data() { return mydata; };
  const T *data() const { return mydata; };

  void zerodata() {
    std::memset(mydata, '\0', size);
  }

  void fakedata() { DataHandler<T>(impl).fake(mydata, size / sizeof(T)); }
  void verifydata() {
    DataHandler<T>(impl).verify(mydata, size / sizeof(T), dims, name);
  }



  // TODO: realloc
private:
  dyn_array(BenchmarkRun *impl, int count, bool verify, std::vector<size_t> dims, std::string_view name) : dyn_array_base(impl, count * sizeof(T), verify, std::move(dims), name), mydata(new T[count]) {}

  // typed, to ensure TBAA can be effective
  T *mydata;
};



template <typename T, size_t DIMS>
class owning_array {
public:
  owning_array(BenchmarkRun *impl, bool verify, typename _make_tuple<typename _make_dimlengths<DIMS>::type>::type dims, std::string_view name)
      : mydata(impl, get_stride(dims), verify, t2v<size_t, DIMS>(dims), name), sizes(dims) {}


  multarray<T, DIMS> get() {
    return multarray<T, DIMS>(mydata.data(), sizes);
  }

  void fakedata() {
    mydata.fakedata();
  }
  void zerodata() {
    mydata.zerodata();
  }

  operator multarray<T, DIMS>() { return get(); }

  //  template<  typename  std::enable_if<DIMS==1,bool> ::type =  true >
  //  operator T*() {  return  mydata.data(); }

  // Implicit conversion to pointer only for flat arrays
  template <typename U = T, typename = typename std::enable_if<std::is_same<U, T>::value && DIMS == 1, T>::type>
  operator T *() { return data(); }

  T *data() { return mydata.data(); }


#if ROSETTA_CUDA

#endif

private:
  dyn_array<T> mydata;
  typename _make_tuple<typename _make_dimlengths<DIMS>::type>::type sizes;
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
  Iterator<AutoIteration> begin();
  Iterator<AutoIteration> end();

  Range manual() { return Range(*this); }

  template <typename T>
  struct InternalData {
    size_t size;
    T Data[];
  };

  // TODO: return some smart ptr, we are C++ after all
  // TODO: ensure alignment
  template <typename T>
  T *malloc(size_t count) {
    auto result = (InternalData<T> *)::malloc(sizeof(InternalData<T>) + sizeof(T) * count);
    result->size = count * sizeof(T);
    addAllocatedBytes(count * sizeof(T));
    return &result->Data[0];
  }

  template <typename T>
  dyn_array<T> alloc_array(size_t count, bool verify = false) {
    return dyn_array<T>(impl, count, verify);
  }

  template <typename T>
  dyn_array<T> calloc_array(size_t count, bool verify = false) {
    auto result = dyn_array<T>(impl, count, verify);
    result.zerodata();
    return result;
  }



  template <typename T, size_t DIMS>
  owning_array<T, DIMS>
  allocate_array(
      typename _make_tuple<typename _make_dimlengths<DIMS>::type>::type sizes,
      bool fakedata,
      bool verify,
      std::string_view name = std::string_view()) {
    owning_array<T, DIMS> result(impl, verify, sizes, name);
    if (fakedata)
      result.fakedata();
    else
      result.zerodata();
    return result; // NRVO
  }

  // FIXME: Template deduction doesn't work on number of elements in @p sizes.
  template <typename T>
  owning_array<T, 1>
  allocate_array(typename _make_tuple<typename _make_dimlengths<1>::type>::type sizes, bool fakedata, bool verify, std::string_view name = std::string_view()) {
    return allocate_array<T, 1>(sizes, fakedata, verify, name);
  }


  template <typename T>
  owning_array<T, 2>
  allocate_array(typename _make_tuple<typename _make_dimlengths<2>::type>::type sizes, bool fakedata, bool verify, std::string_view name = std::string_view()) {
    return allocate_array<T, 2>(sizes, fakedata, verify, name);
  }

  template <typename T>
  owning_array<T, 3>
  allocate_array(typename _make_tuple<typename _make_dimlengths<3>::type>::type sizes, bool fakedata, bool verify, std::string_view name = std::string_view()) {
    return allocate_array<T, 3>(sizes, fakedata, verify, name);
  }



  template <typename T>
  void fakedata(T *data, size_t count) {
    DataHandler<T>(impl).fake(data, count);
  }


  template <typename T>
  void verifydata(T *data, size_t count) {
#if ROSETTA_VERIFY
    DataHandler<T>(impl).verify(data, count);
#else
    // Don't do anything in benchmark mode
#endif
  }

  template <typename T>
  void free(T *ptr) {
    auto internaldata = (InternalData<T> *)(((char *)ptr) - offsetof(InternalData<T>, Data));

    subAllocatedBytes(internaldata->size);
    ::free(internaldata);
  }



#ifdef ROSETTA_PPM_CUDA
  template <typename T>
  T *allocate_dev(size_t n) {
    T *devptr = nullptr;
    // TODO: Count device memory allocation
    BENCH_CUDA_TRY(cudaMalloc((void **)&devptr, n * sizeof(real)));
    return devptr;
  }

  template <typename T>
  void free_dev(T *dev_ptr) {
    BENCH_CUDA_TRY(cudaFree(dev_ptr));
  }

#endif


private:
  State(BenchmarkRun *impl) : impl(impl) {}
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

public:
  typedef std::forward_iterator_tag iterator_category;
  typedef I value_type;
  typedef I &reference;
  typedef I *pointer;
  typedef std::ptrdiff_t difference_type;


  BENCHMARK_ALWAYS_INLINE
  I operator*() const {
    //  printf("operator*()\n");
    return I(state);
  }

  BENCHMARK_ALWAYS_INLINE
  Iterator &operator++() {
    assert(remaining > 0);
    remaining -= 1;
    return *this;
  }

  BENCHMARK_ALWAYS_INLINE
  bool operator!=(Iterator const &that) const {
    if (BENCHMARK_BUILTIN_EXPECT(remaining != 0, true))
      return true;
    remaining = state.refresh();
    assert(remaining >= 0);
    return remaining != 0;
  }

private:
  explicit Iterator(State &state, bool IsEnd) : state(state), isEnd(IsEnd) {}

  State &state;
  mutable int remaining = 0;
  bool isEnd;
};



inline Iterator<Iteration> Range::begin() { return Iterator<Iteration>(state, false); }
inline Iterator<Iteration> Range::end() { return Iterator<Iteration>(state, true); }


inline Iterator<AutoIteration> State::begin() { return Iterator<AutoIteration>(*this, false); }
inline Iterator<AutoIteration> State::end() { return Iterator<AutoIteration>(*this, true); }



#endif /* ROSETTA_H_ */
