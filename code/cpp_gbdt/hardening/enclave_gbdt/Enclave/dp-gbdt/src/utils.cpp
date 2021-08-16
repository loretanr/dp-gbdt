#include <algorithm>
#include <cmath>
#include <numeric>
#include <mutex>
#include "utils.h"

#include <sgx_trts.h>   /* sgx_read_rand */


/** Methods */

int sgx_random_int()
{
    uint16_t rval; 
    sgx_read_rand((unsigned char *) &rval, sizeof(uint16_t));
    return (int) rval;
}

// create some default parameters for quick testing
ModelParams create_default_params()
{
    ModelParams params;
    params.nb_trees = 50;
    params.use_dp = true;
    params.max_depth = 6;
    params.gradient_filtering = true;
    params.balance_partition = true;
    params.leaf_clipping = true;
    params.privacy_budget = 0.1;
    return params;
};

// put a value between two bounds, not in std::algorithm in c++11
double clamp(double n, double lower, double upper)
{
  return std::max(lower, std::min(n, upper));
}


// TODO formula
double log_sum_exp(std::vector<double> vec)
{
    size_t count = vec.size();
    if (count > 0) {
        double maxVal = *std::max_element(vec.begin(), vec.end());
        double sum = 0;
        for (size_t i = 0; i < count; i++) {
            sum += exp(vec[i] - maxVal);
        }
        return log(sum) + maxVal;
    } else {
        return 0.0;
    }
}

double compute_mean(std::vector<double> &vec)
{
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    return sum / vec.size();
}

double compute_stdev(std::vector<double> &vec, double mean)
{
    double sq_sum = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
    return std::sqrt(sq_sum / vec.size() - mean * mean);
}
