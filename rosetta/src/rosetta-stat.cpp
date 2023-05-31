
#include "rosetta-stat.h"

#include "cdflib.hpp"
#include <cstdint> //added as MAX_INT was not supported by Intel compiler
#include <cassert>
#include <cmath>


#if 0
extern "C" {
    bool  isnan(double x) {
        return std::isnan(x);
    };

    bool  isfinite(double x) {
        return std::isfinite(x);
    };
}
#endif



using namespace rosetta;



double Statistic::stddev() {
  return std::sqrt(variance());
}



static double t_compute(double ratio, double stddev, double count) {
  assert(count >= 1);

  double p = 1 - (1 - ratio) / 2; // Two-sided, symmetric
  double scale = stddev / std::sqrt(count);

  int which = 2;
  double q = 1 - p;

#if 0
    double t =    stdtri(count-1, p);
#else
  double t;
  double bound;
  int status = -1;
  double df = count - 1;
  cdft(&which, &p, &q, &t, &df, &status, &bound);
  assert(status == 0 && "Expect CDF to succeed");
#endif
  assert(t >= 0 && "spread cannot be negative");

  return t * scale;
}



double Statistic::abserr(double ratio) {
  if (_count < 2)
    return 0; // Spread not defined with just one value

  return t_compute(ratio, this->stddev(), _count);
#if 0
    double stddev = this->stddev();
    double scale  = stddev   / std::sqrt(_count);
    double  p = 1 - (1 - ratio) / 2 ;// Two-sided, symmetric

    int which = 2;
    double q=1-p;
        double t;
    double bound;
    int status=-1;
    double df = _count -1;
    cdft(&which, &p,& q,& t,& df,& status,&bound);

    assert(status == 0 && "Expect CDF to succeed");
    assert(t >= 0 && "spread cannot be negative");
    return t*scale ;
#endif
}



static double t_estimate_count(double abserr, double ratio, double stddev, double prev_est_count) {
  double scale = stddev / std::sqrt(prev_est_count);

  int which = 3;
  double p = 1 - (1 - ratio) / 2; // Two-sided, symmetric
  double q = 1 - p;
  double t = abserr / scale;
  double df;
  int status = -1;
  double bound;
  cdft(&which, &p, &q, &t, &df, &status, &bound);

  // Impossible to fulfil
  if (status == 2)
    return INFINITY;

  assert(status == 0 && "Expect CDF to succeed");
  return df + 1;
}

// abserr =   t(ratio) * stddev / sqrt(n)
// abserr * sqrt(n) = t(ratio) * stddev
// n = ( t(ratio) * stddev  / abserr )^2


size_t Statistic::min_more_samples(double abserr, double ratio) {
  assert(abserr > 0);
  assert(0 < ratio && ratio < 1);
  assert(_count >= 2 && "Need at least 2 samples to get an estimate");

  double stddev = this->stddev();

  double p = 1 - (1 - ratio) / 2; // Two-sided, symmetric
  double approx_n = sqr(ndtri(p) * stddev / abserr);

  // Student-t distribution is more spread out than normal distribution, hence this is a lower bound
  assert(approx_n <= 1 || t_compute(ratio, stddev, approx_n) >= abserr);


  if (approx_n + 0.5 >= SIZE_MAX)
    return SIZE_MAX;

  // Search for the exact probability boundary
  // TODO: Since we are looking for an underapproximation anyway, approx_n could be sufficient
  size_t base = std::llround(std::floor(approx_n));
  size_t lower = base;
  size_t k = 1;


  // If the error is already within limit return 0. Otherwise, return at least 1
  if (lower <= _count) {
    if (t_compute(ratio, stddev, _count) <= abserr)
      return 0;
    base = _count;
    lower = _count + 1;
    k = 2;
  }



  size_t upper;
  for (;; k *= 2) {
    upper = base + k;
    if (t_compute(ratio, stddev, upper) < abserr) {
      upper -= 1;
      break;
    }

    // Some cutoff
    if (k >= SIZE_MAX / 2)
      return SIZE_MAX;

    lower = upper;
  }



  while (true) {
    if (lower == upper)
      return lower - _count;

    size_t mid = (lower + upper + 1) / 2; // FIXME: possible overflow
    double t = t_compute(ratio, stddev, mid);

    if (t < abserr) {
      // mid is too large, look for something smaller
      upper = mid - 1;
      continue;
    }

    //
    lower = mid;
  }

#if 0

    double  first_estimate_f =  std::ceil( t_estimate_count(abserr,ratio, stddev, _count));
    assert(first_estimate_f >= 0);
    if (first_estimate_f +0.5 >= SIZE_MAX)
        return SIZE_MAX;
    size_t first_estimate =std::max<long long>(  std::llround(first_estimate_f),_count );
  //  if (first_estimate <= _count)
  //      return 0; // Target already reached

    double  second_estimate_f =  std::ceil( t_estimate_count(abserr,ratio, stddev, first_estimate));
    if (second_estimate_f +0.5 >= SIZE_MAX)
        return SIZE_MAX;
    size_t second_estimate=  std::max<long long>(  std::llround(second_estimate_f), _count);


    assert(first_estimate >= second_estimate && "Since first_estimate is larger than _count, the Student-t density curve is more centered, and therefore fewer samples should be needed");



    // Bisection
    size_t lower = second_estimate;
    size_t upper = first_estimate;
    while (true) {
        if (lower  == upper)
            return lower - _count;

        size_t mid = (lower + upper) / 2; // FIXME: possible overflow
        double t = t_compute(ratio, stddev,mid );

        if ( t < abserr ) {
            // mid does fulfill the requirement, but maybe there is a smaller
            upper  = mid;
            continue;
        }

        // forward progress guarantee
        lower = mid +1 ; 
    }



#endif


#if 0
    // Fixpoint iteration because the scale itself depends on n.
    size_t minest = _count;
    size_t maxest;
    while (true) {
        double scale  = stddev   /std:: sqrt(minest);

        int which = 3;
        double p =  1 - (1 - ratio) / 2 ;// Two-sided, symmetric
        double q =1-p ;
        double t = abserr/scale;
        double df ;
        int status=-1 ;
        double bound ;
        cdft(&which,  &p,& q,& t,& df,& status,&bound);
        assert(status == 0 && "Expect CDF to succeed");


        int newest =   std::lround ( std::ceil( df  + 1 ));

        // Fewer samples would have done as well
        // This should only be possible in the first iteration, otherwise we overshot the target
        if (newest < minest)
            return minest - _count;
        
        // Fixpoint reached
        if (minest == newest )
            return newest -  _count;    

        // Pessimize required samples for next iteration
        minest = newest;
    }







    double scale  = stddev   / sqrt(_count);

    int which = 3;
    double p =  1 - (1 - ratio) / 2 ;// Two-sided, symmetric
    double q =1-p ;
    double t = abserr/scale;
    double df ;
    int status=-1 ;
    double bound ;
    cdft(&which,  &p,& q,& t,& df,& status,&bound);
    assert(status == 0 && "Expect CDF to succeed");

    // Goal already reached
    if (df <= 0)
        return 0;

  auto result = std::ceil(df ) - _count + 1;
  if (result <= 0)
      return 0;
  if (result >= INT_MAX)
      return INT_MAX;
  return std::lround ( result);
#endif
}
