#ifndef ROSETTA_H_
#define ROSETTA_H_

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
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


#define BENCH_CUDA_TRY(call)                                 \
  do {                                                       \
    auto const status = (call);                              \
    if (cudaSuccess != status) {                             \
      printf("CUDA call '" #call "' returned %d\n", status); \
      abort();                                               \
    }                                                        \
  } while (0);


using benchmark::ClobberMemory;

class Scope;
class Iteration;
class AutoIteration;
template <typename I>
class Iterator;
class Range;
class State;



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


#pragma region _make_index_sequence
// Based on http://stackoverflow.com/a/6454211
// template<int...> struct index_tuple{};
template <size_t...>
class _indices {};

// template<std::size_t I, typename IndexTuple, typename... Types>
// struct make_indices_impl;
template <std::size_t I /*counter*/, typename IndexTuple /*the index list in construction*/, typename... Types>
struct _index_sequence;

// template<std::size_t I, std::size_t... Indices, typename T, typename... Types>
// struct make_indices_impl<I, index_tuple<Indices...>, T, Types...>
//{
//    typedef typename make_indices_impl<I + 1, index_tuple<Indices...,
// I>, Types...>::type type;
// };
template <std::size_t I, std::size_t... Indices, typename T /*unqueue*/, typename... Types /*remaining in queue*/>
struct _index_sequence<I, _indices<Indices...>, T, Types...> {
  typedef typename /*recursion call*/ _index_sequence<I + 1, _indices<Indices..., I>, Types...>::type type;
};

// template<std::size_t I, std::size_t... Indices>
// struct make_indices_impl<I, index_tuple<Indices...> >
//{
//    typedef index_tuple<Indices...> type;
// };
template <std::size_t I, std::size_t... Indices>
struct _index_sequence<I, _indices<Indices...>> {
  typedef _indices<Indices...> type; // recursion terminator
};

// template<typename... Types>
// struct make_indices : make_indices_impl<0, index_tuple<>, Types...>
//{};
template <typename... Types>
// struct _make_index_sequence : _index_sequence<0, _indices<>, Types...> {}; // Using inheritance because there are no templated typedefs (outside of classes)
using _make_index_sequence = _index_sequence<0, _indices<>, Types...>;
// TODO: Also a version that converts constants parameter packs (instead of type parameter pack)
#pragma endregion



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


template <typename TUPLE, size_t... Is>
static auto extract_subtuple(TUPLE tuple, std::index_sequence<Is...>) {
  return std::make_tuple(std::get<Is>(tuple)...);
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



#if 0
template<typename DIMLENGTHS>
struct  _flatindex {
    using TupleTy = typename  _make_tuple<DIMLENGTHS>::type;
    using RestDimlengths = typename _unqueue_dimlengths< DIMLENGTHS>::rest;

    static ssize_t flatindex(TupleTy lengths, TupleTy coords) {
        auto sublengths = extract_subtuple(lengths, std::make_index_sequence<std::tuple_size<RestDimlengths> > {});
        auto subcoords  = extract_subtuple(coords, std::make_index_sequence<std::tuple_size<RestDimlengths> > {});

        auto len   = get<0>(lengths);
        auto coord = get<0>( coords);

        auto result = _flatindex<RestDimlengths>::flatindex();
        result *= get<0>(lengths);
        result += 
    }
};
#endif



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

#if 0
private:
    template<size_t... Indices>
    subty buildSubtyHelper(_indices<Indices...>/*unused*/, int64_t coords[sizeof...(Indices)], int64_t appendCoord)   {
        return subty(owner, coords[Indices]..., appendCoord);
    }
#endif

public:
  subty operator[](int64_t i) /*TODO: const*/ {
    auto len = _unqueue_tuple<RemainingLengthsTy>::get_first(remainingLengths);
    assert(0 <= i);
    assert(i < len);

    auto rest = _unqueue_tuple<RemainingLengthsTy>::get_rest(remainingLengths);
    return _multarray_partial_subscript<T, nTogo - 1>(data + get_stride(rest), rest);
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


#if 0
    /* constexpr */ int length(int d) const { 
        return DimLengths[d];
    }
#endif


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
    return _multarray_partial_subscript<T, DIMS - 1>(data + get_stride(rest), rest);
  }



#if 0
    /// Overload for length(int) for !D
    template<typename Dummy = void>
    typename std::enable_if<(sizeof...(__L)==1), typename std::conditional<true, int, Dummy>::type >::type
        /* constexpr */  length() { 
        return length(0);
    }
#endif



#if 0
    T *ptr( std::array<size_t, DIMS> coords)  { 
        ssize_t flatindex = coords[0];
        for (int i = 1; i < DIMS; ++i) {
            flatindex *= DimLengths[i];
            flatindex += coords[i];
        }
        return localdata + flatindex;
    }


    T *ptr(size_t coords[DIMS])  { 
        ssize_t flatindex = coords[0];
        for (int i = 1; i < DIMS; ++i) {
            flatindex *= DimLengths[i];
            flatindex += coords[i];
        }
        return localdata + flatindex;
    }




    T *ptr(typename _make_tuple<__L>::type coords)  { 
        ssize_t flatindex = coords[0];
        for (int i = 1; i < DIMS; ++i) {
            flatindex *= DimLengths[i];
            flatindex +=  coords[i];
        }
        return localdata + flatindex;
    }
#endif


#if 0
    template<typename Dummy = void>
    typename std::enable_if<std::is_same<Dummy, void>::value && (DIMS==1), T&>::type
        operator[](int64_t i)  { 
        return *ptr(i);
    }

    typedef _multarray_partial_subscript<T, typename _unqueue<__L...>::first, typename _unqueue<__L...>::rest> subty;


    template<typename Dummy = void>
    typename std::enable_if<std::is_same<Dummy, void>::value && (DIMS>1), subty>::type
        operator[](int64_t i) {
        return subty(this, i);
    }
#endif
}; // class multarray



enum Measure {
  WallTime,
  UserTime,
  KernelTime, // TODO:
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
  void verify(double *data, ssize_t count, std::vector <size_t> dims, std::string_view name);
};



class dyn_array_base {
protected:
  dyn_array_base(BenchmarkRun *impl, int size, bool verify, std::vector<size_t> dims , std::string_view name);
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
    
      //dims.reserve(DIMS);

    DataHandler<T>(impl).verify(mydata, size / sizeof(T), dims, name );
  }



  // TODO: realloc
private:
  dyn_array(BenchmarkRun *impl, int count, bool verify, std::vector<size_t> dims , std::string_view name) : dyn_array_base(impl, count * sizeof(T), verify,std::move(dims), name), mydata(new T[count]) {}

  // typed, to ensure TBAA can be effective
  T *mydata;
};



template <typename T, size_t DIMS>
class owning_array {
public:
  owning_array(BenchmarkRun *impl, typename _make_tuple<typename _make_dimlengths<DIMS>::type>::type sizes, bool verify, std::vector<size_t> dims , std::string_view name) : mydata(impl, get_stride(sizes), verify, std::move( dims), name), sizes(sizes) {}

  multarray<T, DIMS> get() {
    return multarray<T, DIMS>(mydata.data(), sizes);
  }


  operator multarray<T, DIMS>() { return get(); }

  //  template<  typename  std::enable_if<DIMS==1,bool> ::type =  true >
  //  operator T*() {  return  mydata.data(); }

  // Implicit conversion to pointer only for flat arrays
  template <typename U = T, typename = typename std::enable_if<std::is_same<U, T>::value && DIMS == 1, T>::type>
  operator T *() { return mydata.data(); }

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

#if 0
  template <typename T>
  dyn_array<T> fakedata_array(size_t count, bool verify = false) {
    auto result = dyn_array<T>(impl, count, verify);
    result.fakedata();
    return result;
  }
#endif 


#if 0
    template<typename T, size_t DIMS> 
    owning_array< T,DIMS > 
        allocate_array(  std::array<size_t,DIMS> sizes, bool fakedata, bool verify   ) {
        auto  result =  dyn_array<T>( impl, get_stride(sizes) ,verify );
        if (fakedata)
            result.fakedata();
        else 
            result.zerodata();
        return    owning_array< T,DIMS> (  impl, sizes,verify );
    }


#else
  // TODO: Generalize for arbitrary array sizes

  template <typename T>
  owning_array<T, 1u>
  allocate_array(typename _make_tuple<typename _make_dimlengths<1>::type>::type sizes, bool fakedata, bool verify, std::string_view name = std::string_view()) {
      std::vector<size_t> dims{(size_t)std::get<0>(sizes)};
      auto result = dyn_array<T>(impl, get_stride(sizes), verify,dims , name);
    if (fakedata)
      result.fakedata();
    else
      result.zerodata();
    return owning_array<T, 1u>(impl, sizes, verify, dims, name );
  }

  template <typename T>
  owning_array<T, 2u>
  allocate_array(typename _make_tuple<typename _make_dimlengths<2>::type>::type sizes, bool fakedata, bool verify, std::string_view name = std::string_view()) {
      std::vector<size_t>dims{(size_t)std::get<0>(sizes),(size_t)std:: get<1>(sizes)};
      auto result = dyn_array<T>(impl, get_stride(sizes), verify,dims , name);
    if (fakedata)
      result.fakedata();
    else
      result.zerodata();
    return owning_array<T, 2u>(impl, sizes, verify, dims,name);
  }

  template <typename T>
  owning_array<T, 3u>
  allocate_array(typename _make_tuple<typename _make_dimlengths<3>::type>::type sizes, bool fakedata, bool verify, std::string_view name = std::string_view()) {
      std::vector<size_t> dims{(size_t)std::get<0>(sizes), (size_t)std::get<1>(sizes), (size_t)std::get<2>(sizes)};
      auto result = dyn_array<T>(impl, get_stride(sizes), verify, dims, name);
    if (fakedata)
      result.fakedata();
    else
      result.zerodata();
    return owning_array<T, 3u>(impl, sizes, verify, dims,name);
  }
#endif



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


#ifdef ROSETTA_REALTYPE
typedef ROSETTA_REALTYPE real;
#endif



#if 0


template<typename T/*Elt type*/, typename Stored/*coordinates already known*/, typename Togo/*coordinates to go*/ >
class _array_partial_subscript; // Never instantiate, use specializations only 

template<typename T, int... Stored, int... Togo> 
class _array_partial_subscript<T, _dimlengths<Stored...>, _dimlengths<Togo...>> {
    static_assert(sizeof...(Togo) > 1, "Specialization for more-coordinates-to-come");

    static const int nStored = sizeof...(Stored);
    static const int nTogo = sizeof...(Togo);
    static const int nDim = nStored + nTogo;

    typedef array<T, Stored..., Togo...> fieldty;
    typedef _array_partial_subscript<T, _dimlengths<Stored..., _unqueue<Togo...>::value>, typename _unqueue<Togo...>::rest > subty;

    fieldty *owner;
    int64_t coords[nStored];
public:
    _array_partial_subscript(fieldty *owner, typename _inttype<Stored>::type... coords)  
        : owner(owner), coords({coords...}) {
        //assert(owner);
        //TODO: assertion that all stored are in range
    }

private:
    template<size_t... Indices>
    subty buildSubtyHelper(_indices<Indices...>/*unused*/, int64_t coords[sizeof...(Indices)], int64_t appendCoord)   {
        return subty(owner, coords[Indices]..., appendCoord);
    }

public:
    subty operator[](int64_t i) /*TODO: const*/ {
        //assert(0 <= i);
        //assert(i < _unqueue<Togo...>::value);
        return buildSubtyHelper(typename _make_index_sequence<typename _inttype<Stored>::type...>::type(), coords, i);
    }
}; // class _array_partial_subscript




template<typename T, int Togo, int64_t... Stored>
class _array_partial_subscript<T, _dimlengths<Stored...>, _dimlengths<Togo>> {
    static const int nStored = sizeof...(Stored);
    static const int nTogo = 1;
    static const int nDim = nStored + nTogo;

    typedef array<T, Stored..., Togo> fieldty;

    fieldty *owner;
    int64_t coords[nStored];
public:
   _array_partial_subscript(fieldty *owner, typename _inttype<Stored>::type... coords) 
        : owner(owner), coords({coords...}) {
        //uint64_t tmp[] = {coords...};
        //std::copy(&tmp[0], &tmp[sizeof...(coords)], this->coords);
        //assert(owner);
    }

private:
    template<size_t... Indices>
     T &getPtrHelper(_indices<Indices...>/*unused*/, int64_t coords[sizeof...(Indices)], int64_t last)  {
        return *owner->ptr(coords[Indices]..., last);
    }

public:
     T &operator[](int64_t i)  /*TODO: const*/{
        //assert(0 <= i); // Check lower bound of coordinate
        //assert(i < Togo); // Check upper bound of coordinate
        return getPtrHelper(typename _make_index_sequence<typename _inttype<Stored>::type...>::type(), this->coords, i);
        //return *owner->ptr(coords[_indices<Stored>]..., i);
    }
}; // class _array_partial_subscript





template<int> struct AsInt {
    typedef int type;
};









template<typename ... Args> 
class _consume_parameterpack {
public:
    typedef std::true_type type;
    static const bool value = true;
};

template<typename First, typename ... Args> 
class _first_parampack {
public:
    typedef First type;
};

template<bool First, typename ... Args> 
class _condition {
public:
    typedef std::integral_constant<bool, First> type;
    static const bool value = First;
};



template<size_t i, int Front, int... Values>
class _getval {
public:
    static const int value = _getval<i-1,Values...>::value;
};

template<int Front, int...Values>
class _getval<0, Front, Values...> {
public:
    static const int value = Front;
};

#pragma region _select
static inline int _select(int i); // Forward declaration

template<typename FirstType, typename... Types>
static inline int _select(int i, FirstType first, Types... list)  {
    //assert(i < (int)(1/*first*/+sizeof...(list))); // Interferes with natural loop detection
    if (i == 0)
        return first;
    return _select(i-1, list...); // This is no recursion, every _select has a different signature
}


static inline int _select(int i) { // overload for compiler-time termination, should never be called
#ifdef _MSC_VER
    __assume(false);
#else
    __builtin_unreachable();
#endif
}
#pragma endregion



#if 0
extern int dummyint;
#endif 


template<typename T, uint64_t Dims>
class  field {
public:
}; // class field


#define LLVM_OVERRIDE override

#if 0
#pragma region LocalStore
class LocalStore {
public:
    LocalStore() {  }
    virtual ~LocalStore() {  }

    virtual void init(uint64_t countElts) = 0;
    virtual void release() = 0;

    virtual void *getDataPtr() const = 0; //TODO: Unvirtualize for speed
    virtual size_t getElementSize() const = 0;
    virtual uint64_t getCountElements() const = 0;
}; // class LocalStore
#pragma endregion
#endif

#define MOLLY_DEBUG_METHOD_ARGS(...)
#define MOLLY_DEBUG_FUNCTION_SCOPE

   //#ifndef __MOLLYRT
#if 1
   /// A multi-dimensional array; the dimensions must be given at compile-time
   /// T = underlaying type (must be POD)
   /// __L = sizes of dimensions (each >= 1)
   // TODO: Support sizeof...(__L)==0
template<typename T, int... __L>
class  array: public LocalStore, public field<T, sizeof...(__L)> {



#pragma region LocalStore
private:
    size_t localelts;
    T *localdata;

    void init(uint64_t countElts) LLVM_OVERRIDE{ MOLLY_DEBUG_METHOD_ARGS(countElts, sizeof(T))
        assert(!localdata && "No double-initialization");
    localdata = new T[countElts];
    localelts = countElts;
    }

    void release() LLVM_OVERRIDE { MOLLY_DEBUG_FUNCTION_SCOPE
        delete localdata;
    localdata = nullptr;
    }

    void *getDataPtr() const LLVM_OVERRIDE { MOLLY_DEBUG_FUNCTION_SCOPE
        return localdata;
    }

    size_t getElementSize() const LLVM_OVERRIDE { MOLLY_DEBUG_FUNCTION_SCOPE
        return sizeof(T);
    }

    uint64_t getCountElements() const LLVM_OVERRIDE { MOLLY_DEBUG_FUNCTION_SCOPE
        return localelts;
    }
#pragma endregion


#define LLVM_ATTRIBUTE_USED

#ifndef NDEBUG
    // Allow access to assert, etc.
public:
#else
private:
#endif

    uint64_t coords2idx(typename _inttype<__L>::type... coords) const ;
    //{
    // return __builtin_molly_local_indexof(this, coords...);
    //}
#if 0
    size_t coords2idx(typename _inttype<__L>::type... coords) const MOLLYATTR(fieldmember) { MOLLY_DEBUG_FUNCTION_SCOPE
        MOLLY_DEBUG("coords2idx(" << out_parampack(", ", coords...) << ")");

    assert(__builtin_molly_islocal(this, coords...));
    return __builtin_molly_local_indexof(this, coords...);

#if 0
    size_t idx = 0;
    size_t lastlocallen = 0;
    size_t localelts = 1;
    for (auto d = Dims-Dims; d<Dims; d+=1) {
        auto len = _select(d, __L...);
        auto locallen = __builtin_molly_locallength(this, (uint64_t)d);
        auto coord = _select(d, coords...);
        auto clustercoord = cart_self_coord(d);
        auto localcoord = coord - clustercoord*locallen;
        MOLLY_DEBUG("d="<<d << " len="<<len << " locallen="<<locallen << " coord="<<coord << " clustercoord="<<clustercoord << " localcoord="<<localcoord);
        assert(0 <= localcoord && localcoord < locallen);
        idx = idx*lastlocallen + localcoord;
        lastlocallen = locallen;
        localelts *= locallen;
    }
    MOLLY_DEBUG("this->localelts="<<this->localelts << " localelts="<<localelts);
    assert(this->localelts == localelts);
    assert(0 <= idx && idx < localelts);
    MOLLY_DEBUG("RETURN coords2idx(" << out_parampack(", ", coords...) << ") = " << idx);
    return idx;
#endif
    }
#endif


     uint64_t coords2rank(typename _inttype<__L>::type... coords) const;
    //{
    //  return __builtin_molly_rankof(this, coords...);
    //}
#if 0
    /// Compute the rank which stores a specific value
    rank_t coords2rank(typename _inttype<__L>::type... coords) const MOLLYATTR(fieldmember) { MOLLY_DEBUG_FUNCTION_SCOPE
        MOLLY_DEBUG("coords2rank(" << out_parampack(", ", coords...) << ")");

    return __builtin_molly_rankof(this, coords...);
#if 0
    rank_t rank = 0;
    for (auto d = Dims-Dims; d<Dims; d+=1) {
        auto len = _select(d, __L...);
        auto locallen = __builtin_molly_locallength(this, (uint64_t)d);
        auto coord = _select(d, coords...);
        auto clustercoord = coord / locallen;
        auto clusterlen = (len / locallen) + 1;
        assert(clustercoord < clusterlen);
        rank = (rank * clusterlen) + clustercoord;
    }
    return rank;
#endif
    }
#endif


    LLVM_ATTRIBUTE_USED bool isLocal(typename _inttype<__L>::type... coords) const  {      
    auto expectedRank = coords2rank(coords...);
    auto myrank = __molly_cluster_myrank();
    //world_self(); // TODO: This is the MPI rank, probably not the same as what molly thinks the rank is
    return expectedRank==myrank;

#if 0
    auto expRank = coords2rank(coords...);
    for (auto d = Dims-Dims; d<Dims; d+=1) {
        auto len = _select(d, __L...);
        auto locallen = __builtin_molly_locallength(this, (uint64_t)d);
        auto coord = _select(d, coords...);
        auto clustercoord = cart_self_coord(d); //TODO: Make sure this is inlined
                                                //MOLLY_DEBUG("d="<<d << " len="<<len << " locallen="<<locallen << " coord="<<coord << " clustercoord="<<clustercoord);

        auto localbegin = clustercoord*locallen;
        auto localend = localbegin+locallen;
        MOLLY_VAR(d,len,locallen,coord,clustercoord,localbegin,localend);
        if (localbegin <= coord && coord < localend)
            continue;

        MOLLY_DEBUG("rtn=false coords2rank(coords...)="<<expRank<< " self="<<world_self());
        assert(expRank != world_self());
        return false;

    }
    MOLLY_DEBUG("rtn=true coords2rank(coords...)="<<expRank<< " self="<<world_self());
    assert(expRank == world_self());
    return true;
#endif
    }

public:
    typedef T ElementType;
    static const auto Dims = sizeof...(__L);


    ~array() { MOLLY_DEBUG_FUNCTION_SCOPE
        __builtin_molly_field_free(this);
    delete[] localdata;
    }


    array() { MOLLY_DEBUG_FUNCTION_SCOPE
        MOLLY_DEBUG("array dimension is (" << out_parampack(", ", __L...) << ")");

    //TODO: Do not call here, Molly should generate a call to __molly_field_init for every field it found
    //__builtin_molly_field_init(this); // inlining is crucial since we need the original reference to the field in the first argument
    //EDIT: Now inserted by compiler magic
    //FIXME: Relooking at the source, only to those that have a #pragma transform????

#if 0
    localelts = 1;
    for (auto d = Dims-Dims; d < Dims; d+=1) {
        auto locallen = __builtin_molly_locallength(this, (uint64_t)d);
        MOLLY_DEBUG("__builtin_molly_locallength(this, "<<d<<")=" << locallen);
        localelts *= locallen;
    }
    MOLLY_DEBUG("localelts=" << localelts);
    localdata = new T[localelts];
    assert(localdata);

    if (std::getenv("bla")==(char*)-1) {
        MOLLY_DEBUG("This should never execute");
        // Dead code, but do not optimize away so the template functions get instantiated
        //TODO: Modify clang::CodeGen to generate the unconditionally
        T dummy;
        (void)ptr(static_cast<int>(__L)...);
        (void)__get_local(dummy, static_cast<int>(__L)...);
        (void)__set_local(dummy, static_cast<int>(__L)...);
        (void)__ptr_local(static_cast<int>(__L)...);
        (void)__get_broadcast(dummy, static_cast<int>(__L)...);
        (void)__set_broadcast(dummy, static_cast<int>(__L)...);
        (void)__get_master(dummy, static_cast<int>(__L)...);
        (void)__set_master(dummy, static_cast<int>(__L)...);
        (void)isLocal(__L...);
    }
#endif

    if (1==0) {
        MOLLY_DEBUG("This should never execute");
        // Dead code, but do not optimize away so the template functions get instantiated
        //TODO: Modify clang::CodeGen to generate the unconditionally
        T dummy;
        (void)ptr(static_cast<int>(__L)...);
        (void)__get_local(dummy, static_cast<int>(__L)...);
        (void)__set_local(dummy, static_cast<int>(__L)...);
        (void)__ptr_local(static_cast<int>(__L)...);
        (void)__get_broadcast(dummy, static_cast<int>(__L)...);
        (void)__set_broadcast(dummy, static_cast<int>(__L)...);
        (void)__get_master(dummy, static_cast<int>(__L)...);
        (void)__set_master(dummy, static_cast<int>(__L)...);
        (void)isLocal(__L...);
    }
    }


    /* constexpr */ int length(uint64_t d) const { //MOLLY_DEBUG_FUNCTION_SCOPE
                                                                                                                     //assert(0 <= d && d < (int)sizeof...(__L));
        return _select(d, __L...);
    }

    /// Overload for length(int) for !D
    template<typename Dummy = void>
    typename std::enable_if<(sizeof...(__L)==1), typename std::conditional<true, int, Dummy>::type >::type
        /* constexpr */  length() { //MOLLY_DEBUG_FUNCTION_SCOPE
        return length(0);
    }


    /// Returns a pointer to the element with the given coordinates; Molly will track loads and stores to this memory location and insert communication code
    T *ptr(typename _inttype<__L>::type... coords)  { //MOLLY_DEBUG_FUNCTION_SCOPE
        return (T*)__builtin_molly_ptr(this, coords...);
    }


    template<typename Dummy = void>
    typename std::enable_if<std::is_same<Dummy, void>::value && (sizeof...(__L)==1), T&>::type
         operator[](int64_t i)  { //MOLLY_DEBUG_FUNCTION_SCOPE
                                                                          //assert(0 <= i);
                                                                          //assert(i < _unqueue<__L...>::value);
        return *ptr(i);
    }

    typedef _array_partial_subscript<T, typename _unqueue<__L...>::first, typename _unqueue<__L...>::rest> subty;

    template<typename Dummy = void>
    typename std::enable_if<std::is_same<Dummy, void>::value && (sizeof...(__L)>1), subty>::type
       operator[](int64_t i) { //MOLLY_DEBUG_FUNCTION_SCOPE
                                                                         //assert(0 <= i);
                                                                         //assert(i < _unqueue<__L...>::value);
        return subty(this, i);
    }

#if 0
#pragma region Local access
    LLVM_ATTRIBUTE_USED void __get_local(T &val, typename _inttype<__L>::type... coords) const  { MOLLY_DEBUG_FUNCTION_SCOPE
        //assert(__builtin_molly_islocal(this, coords...));
        auto idx = coords2idx(coords...);
    assert(0 <= idx && idx < localelts);
    assert(localdata);
    val = localdata[idx];
    }
    LLVM_ATTRIBUTE_USED void __set_local(const T &val, typename _inttype<__L>::type... coords) { MOLLY_DEBUG_FUNCTION_SCOPE
        //assert(__builtin_molly_islocal(this, coords...));
        auto idx = coords2idx(coords...);
    assert(0 <= idx && idx < localelts);
    assert(localdata);
    localdata[idx] = val;
    }
    LLVM_ATTRIBUTE_USED T *__ptr_local(typename _inttype<__L>::type... coords) { MOLLY_DEBUG_FUNCTION_SCOPE
        MOLLY_DEBUG("Coords are (" << out_parampack(", ", coords...) << ")");
    //assert(__builtin_molly_islocal(this, coords...));
    auto idx = coords2idx(coords...);
    assert(0 <= idx && idx < localelts);
    assert(localdata);
    return &localdata[idx];
    }
    const T *__ptr_local(typename _inttype<__L>::type... coords) const {
        return const_cast<T*>(__ptr_local(coords...));
    }
#pragma endregion


    LLVM_ATTRIBUTE_USED void __get_broadcast(T &val, typename _inttype<__L>::type... coords) const ;
    LLVM_ATTRIBUTE_USED void __set_broadcast(const T &val, typename _inttype<__L>::type... coords) { MOLLY_DEBUG_FUNCTION_SCOPE
        MOLLY_DEBUG("coords=("<<out_parampack(", ", coords...) << ") __L=("<<out_parampack(", ", __L...) <<")");
    if (isLocal(coords...)) {
        __set_local(val, coords...);
    } else {
        // Nonlocal value, forget it!
        // In debug mode, we could check whether it is really equal on all ranks
    }
    }

    LLVM_ATTRIBUTE_USED void __get_master(T &val, typename _inttype<__L>::type... coords) const __attribute__((molly_fieldmember)) __attribute__((molly_get_master)) { MOLLY_DEBUG_FUNCTION_SCOPE
    }
    LLVM_ATTRIBUTE_USED void __set_master(const T &val, typename _inttype<__L>::type... coords) const __attribute__((molly_fieldmember)) __attribute__((molly_set_master)) { MOLLY_DEBUG_FUNCTION_SCOPE
    }
#endif 

private:
    uint64_t localoffset(uint64_t d) { return __builtin_molly_localoffset(this, d); }
    uint64_t locallength(uint64_t d) { return __builtin_molly_locallength(this, d); }

} ; // class array
#endif

#endif

#endif /* ROSETTA_H_ */
