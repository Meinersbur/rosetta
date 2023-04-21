#include "rosetta-stat.h"

#include "cdflib.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <climits>
#include <cmath>



using namespace rosetta;
using namespace Catch::Matchers;



TEST_CASE("Student-t (compute interval)", "[cdflib]") {
  int which = 2;
  double p = 0.975;
  double q = 0.025;
  double t = NAN;
  double df = 1;
  int status = INT_MIN;
  double bound = NAN;
  cdft(&which, &p, &q, &t, &df, &status, &bound);

  REQUIRE(status == 0);
  CHECK(bound == 0);
  CHECK(which == 2);
  CHECK(p == 0.975);
  CHECK(q == 0.025);
  CHECK(df == 1);
  CHECK_THAT(t, WithinULP(12.706204736432095, 4));
}



TEST_CASE("Student-t (compute interval)", "[cephes]") {
  double p = 0.975;
  double q = 0.025;
  int df = 1;

  double t = stdtri(df, p);

  CHECK_THAT(t, WithinULP(12.706204736432095, 144903));
}



TEST_CASE("Student-t (compute n) Test 1", "[cdflib]") {
  int which = 3;
  double p = 0.975;
  double q = 0.025;
  double t = 12.706204736432095;
  double df = NAN;
  int status = INT_MIN;
  double bound = NAN;
  cdft(&which, &p, &q, &t, &df, &status, &bound);
  REQUIRE(status == 0);
  CHECK(bound == 0);

  CHECK(which == 3);
  CHECK(p == 0.975);
  CHECK(q == 0.025);
  CHECK(t == 12.706204736432095);
  CHECK_THAT(df, WithinULP(1.0, 143526));
}



TEST_CASE("Student-t (compute n) Test 2", "[cdflib]") {
  int which = 3;
  double p = 0.95;
  double q = 0.05;
  double t = 2;
  double df = NAN;
  int status = INT_MIN;
  double bound = NAN;
  cdft(&which, &p, &q, &t, &df, &status, &bound);
  REQUIRE(status == 0);
  CHECK(bound == 0);

  CHECK(which == 3);
  CHECK(p == 0.95);
  CHECK(q == 0.05);
  CHECK(t == 2);
  CHECK_THAT(df, WithinULP(5.1761356827823253, 143526));
}



TEST_CASE("Student-t (compute p,q)", "[cdflib]") {
  int which = 1;
  double p = NAN;
  double q = NAN;
  double t = 12.706204736432095;
  double df = 1;
  int status = INT_MIN;
  double bound = NAN;
  cdft(&which, &p, &q, &t, &df, &status, &bound);
  REQUIRE(status == 0);
  CHECK(bound == 0);

  CHECK(which == 1);
  CHECK(t == 12.706204736432095);
  CHECK(df == 1);
  CHECK_THAT(p, WithinULP(0.975, 145369));
  CHECK_THAT(q, WithinULP(0.025, 145369));
}



TEST_CASE("Count", "[stat]") {
  CHECK(Statistic({}).count() == 0);
  CHECK(Statistic({0.0}).count() == 1);
  CHECK(Statistic({0, 1}).count() == 2);
}


TEST_CASE("Mean", "[stat]") {
  CHECK(Statistic({42}).mean() == 42);
  CHECK(Statistic({2, 3, 7}).mean() == 4);
}


TEST_CASE("Variance", "[stat]") {
  CHECK(Statistic({}).variance() == 0);
  CHECK(Statistic({42}).variance() == 0);
  CHECK(Statistic({2, 6}).variance() == 4);
}

TEST_CASE("StdDev", "[stat]") {
  CHECK(Statistic({}).stddev() == 0);
  CHECK(Statistic({42}).stddev() == 0);
  CHECK(Statistic({-1, 1}).stddev() == 1);
  CHECK(Statistic({2, 6}).stddev() == 2);
  CHECK(Statistic({0, 0, 6, 6}).stddev() == 3);
  CHECK_THAT(Statistic({0, 0, 0, 0, 3.5, 7, 7, 7, 7}).stddev(), WithinULP(3.2998316455372216, 0));
}



TEST_CASE("Confidence interval", "[stat]") {
  CHECK(Statistic({}).abserr() == 0);
  CHECK(Statistic({42}).abserr() == 0);

  // Constants come from SciPy (which internally also uses cdflib, but its Fortran version)
  // FIXME: Difference of 4 ULP
  CHECK_THAT(Statistic({0, 1}).abserr(), WithinULP(4.492321766137882, 4));
  CHECK_THAT(Statistic({-1, 1}).abserr(), WithinULP(8.984643532275763, 4));
  CHECK_THAT(Statistic({0, 0, 6, 6}).abserr(), WithinULP(4.7736694579263945, 4));
  CHECK_THAT(Statistic({0, 0, 0, 0, 3.5, 7, 7, 7, 7}).abserr(), WithinULP(2.5364751398409346, 4));
}



TEST_CASE("min_more_samples", "[stat]") {
  CHECK(Statistic({0, 1}).min_more_samples(1, 0.99) == 3);
  CHECK(Statistic({0, 1}).min_more_samples(1, 0.95) == 1);
  CHECK(Statistic({0, 1}).min_more_samples(1, 0.9) == 1);
  CHECK(Statistic({0, 1}).min_more_samples(0.1, 0.99) == 167);
  CHECK(Statistic({0, 0.25, 0.5, 0.75, 1}).min_more_samples(0.2, 0.8) == 1);

  // If fewer samples would have done as well
  CHECK(Statistic({0, 0.5, 1}).min_more_samples(4.5, 0.8) == 0);

  // Slightly below and above the target abserr.
  CHECK(Statistic({0, 1}).min_more_samples(4.5, 0.95) == 0);
  CHECK(Statistic({0, 1}).min_more_samples(4.48, 0.95) == 1);

  // Impossibly tight bounds
  CHECK(Statistic({0, 1}).min_more_samples(0.0000000000000001, 0.9999999999999) == SIZE_MAX);
}


TEST_CASE("min_more_samples_rel", "[stat]") {
  // Same as first case of min_more_samples (due to mean == 1)
  CHECK(Statistic({0.5, 1.5}).min_more_samples_rel(1, 0.99) == 3);
}
