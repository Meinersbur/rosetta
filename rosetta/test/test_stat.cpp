#include "rosetta-stat.h"
#include "cdflib.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>


using namespace rosetta;
using namespace  Catch::Matchers;

TEST_CASE("Student-t", "[cdflib]") {
    int which = 2;
    double p =  0.975 ;
    double q =0.025 ;
    double t = NAN;
    double df = 1;
    int status = INT_MIN;
    double bound =NAN;
    cdft(&which,  &p,& q,& t,& df,& status,&bound);

    CHECK(which == 2);
    CHECK(p == 0.975  );
    CHECK(q == 0.025);
   CHECK_THAT(t , WithinULP(        12.706204736432095,4));
    CHECK(df == 1);
    CHECK(status ==0);
    CHECK(bound ==0);
}


TEST_CASE("Count", "[stat]") {
    CHECK(  Statistic({}).count() ==0 );
    CHECK(  Statistic({0.0}).count() ==1 );
    CHECK(  Statistic({0,1}).count() ==2 );
}


TEST_CASE("Confidence interval", "[stat]") {
    // Constant comes from SciPy (which internally also uses cdflib)
    CHECK(Statistic({}).abserr() == 0);
    CHECK(Statistic({42}).abserr() == 0);
    CHECK_THAT(Statistic({0,1}).abserr() , WithinULP(      4.492321766137882,4));
    CHECK_THAT(Statistic({-1,1}).abserr() , WithinULP(  8.984643532275763,4));
    CHECK_THAT(Statistic({0,0,6,6}).abserr() , WithinULP(  4.7736694579263945,4));
    CHECK_THAT(Statistic({0, 0, 0, 0, 3.5, 7, 7, 7, 7}).abserr() , WithinULP( 2.5364751398409346,4));
}

