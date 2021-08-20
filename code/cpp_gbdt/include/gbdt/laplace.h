#ifndef LAPLACE_H
#define LAPLACE_H

#include <random>

/*
    This method for sampling from laplace distribution is described here:
    https://www.johndcook.com/blog/2018/03/13/generating-laplace-random-variables/
    DPBoost also relies on this mechanism:
    https://github.com/QinbinLi/DPBoost/blob/1174730f9b99aca8389c0721fc3864402236e5cd/include/LightGBM/random_generator.h
*/
class Laplace
{
private:
    double scale;
    std::mt19937 generator;
    std::default_random_engine generator1;
    std::default_random_engine generator2;
    std::exponential_distribution<double> distribution;
public:
    Laplace(int seed): generator(seed){};
    Laplace(double _scale, int seed): scale(_scale), generator(seed), distribution(1.0/_scale){};

    double return_a_random_variable()
    {
    double e1 = distribution(generator);
    double e2 = distribution(generator);
    return e1-e2;
    }

    double return_a_random_variable(double _scale)
    {
    std::exponential_distribution<double> distribution1(1.0/_scale);
    std::exponential_distribution<double> distribution2(1.0/_scale);
    double e1 = distribution1(generator);
    double e2 = distribution2(generator);
    return e1-e2;
    }
};

#endif /* LAPLACE_H */
