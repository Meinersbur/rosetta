
#include "rosetta-stat.h"

#include "cdflib.hpp"

#include <cmath>
#include <cassert>

using namespace rosetta;





double Statistic:: stddev() {
    return std::sqrt(variance());
} 


// 95% confidence interval around mean
double  Statistic:: abserr(double ratio ) {
    if (_count <2)
        return 0;  // Spread not defined with just one value

//    auto mean = this->mean();
    auto stddev = this->stddev();
    auto scale  = stddev   / sqrt(_count);



    auto  p = 1 - (1 - ratio) / 2 ;// Two-sided

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
}

