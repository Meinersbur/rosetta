#ifndef ROSETTA_STAT_H
#define ROSETTA_STAT_H

#include <cstddef>
#include <initializer_list>
		

namespace rosetta{ 

    // Avoid double evaluation of v
    static double sqr(double v) {
        return v * v;
    }


    class Statistic {
    private:
        size_t _count ;

        double _sum;
        double _sumsqr ;

    public :
        template <size_t N>
        Statistic(double vals[N]) : Statistic(&vals[0], N) {}

      
        Statistic(std::initializer_list<double >vals) : Statistic(vals.begin(), vals.size()) {}

        explicit Statistic(const double *data, size_t count) : _count( count) {
            double sum = 0;
            double sumsqr = 0;
            for (int i = 0; i < count; ++i) {
                sum += data[i];
                sumsqr += sqr(data[i]);
            }
            this->_sum=sum;
            this->_sumsqr = sumsqr;
        }


        size_t count() {return _count;}

        double mean() {
            return _sum / _count;
        }


        double variance() {
            return _sumsqr  / _count - sqr(mean());
        }


        double stddev() ;


        // 95% confidence interval around mean
        double abserr(double ratio =0.95) ;


        double relerr(double ration = 0.95) {
            return abserr() / mean();
        }

    };




}


#endif /* ROSETTA_STAT_H */
