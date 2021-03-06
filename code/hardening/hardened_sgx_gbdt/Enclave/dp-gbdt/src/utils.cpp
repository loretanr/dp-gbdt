#include <algorithm>
#include <cmath>
#include <numeric>
#include <mutex>
#include "utils.h"
#include "constant_time.h"

#include <sgx_trts.h>   /* sgx_read_rand */


/** Methods */

int sgx_random_pos_int()
{
    int32_t rval;
    sgx_read_rand((unsigned char *) &rval, sizeof(int32_t));
    return std::abs(rval);  // [0 .. 2'147'483'647]
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
    n = constant_time::select(n < lower, lower, n);
    n = constant_time::select(n > upper, upper, n);
    return n;
}

bool double_equality(double d1, double d2)
{
    double epsilon = std::numeric_limits<double>::epsilon();
    double maxd1d2One = std::max( { 1.0, std::fabs(d1) , std::fabs(d2) } ) ;
    return std::fabs(d1 - d2) <= epsilon * maxd1d2One ;
}

double log_sum_exp(std::vector<double> vec)
{
    size_t count = vec.size();
    if (count > 0) {
        double max_val = std::numeric_limits<double>::min();
        for (auto elem : vec) {
            max_val = constant_time::max(max_val, elem);
        }
        double sum = 0;
        for (size_t i = 0; i < count; i++) {
            sum += exp(vec[i] - max_val);
        }
        return log(sum) + max_val;
    } else {
        return 0.0;
    }
}

double compute_mean(std::vector<double> &vec)
{
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    return sum / (double) vec.size();
}

double compute_stdev(std::vector<double> &vec, double mean)
{
    double sq_sum = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
    return std::sqrt(sq_sum / (double) vec.size() - mean * mean);
}
