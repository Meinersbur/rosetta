#ifndef ROSETTA_STAT_H
#define ROSETTA_STAT_H

#include <cstddef>
#include <initializer_list>


extern "C" {
// From cephes library
double stdtr(int k, double t);
double stdtri(int k, double p);
double ndtri(double y0);
double ndtr(double a);
}



namespace rosetta {

// Avoid double evaluation of v
static double sqr(double v) {
  return v * v;
}


class Statistic {
private:
  size_t _count;

  double _sum;
  double _sumsqr;

public:
  template <size_t N>
  Statistic(double vals[N]) : Statistic(&vals[0], N) {}


  Statistic(std::initializer_list<double> vals) : Statistic(vals.begin(), vals.size()) {}

  explicit Statistic(const double *data, size_t count) : _count(count) {
    double sum = 0;
    double sumsqr = 0;
    for (int i = 0; i < count; ++i) {
      sum += data[i];
      sumsqr += sqr(data[i]);
    }
    this->_sum = sum;
    this->_sumsqr = sumsqr;
  }


  size_t count() { return _count; }

  double mean() {
    return _sum / _count;
  }


  double variance() {
    if (_count < 1)
      return 0;
    return _sumsqr / _count - sqr(mean());
  }


  double stddev();


  // Confidence interval around mean
  double abserr(double ratio = 0.95);


  double relerr(double ratio = 0.95) {
    return abserr(ratio) / mean();
  }


  // Estimate required additional samples to get the abserr below the constant.
  size_t min_more_samples(double abserr, double ratio = 0.9);

  size_t min_more_samples_rel(double relerr, double ratio = 0.9) {
    return min_more_samples(relerr * mean(), ratio);
  }
};



} // namespace rosetta


#endif /* ROSETTA_STAT_H */
