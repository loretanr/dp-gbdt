#include <algorithm>
#include <cmath>
#include <numeric>
#include <mutex>
#include "utils.h"


/** Global Variables */

bool VERIFICATION_MODE;
size_t cv_fold_index;


/** Methods */

// create some default parameters for quick testing
ModelParams create_default_params()
{
    ModelParams params;
    params.nb_trees = 50;
    params.max_depth = 6;
    params.gradient_filtering = TRUE;
    params.balance_partition = TRUE;
    params.leaf_clipping = TRUE;
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

std::string get_time_string()
{
    time_t t = time(0);
    struct tm *now = localtime(&t);
    char buffer [80];
    strftime(buffer,80,"%m.%d_%H:%M",now);
    return std::string(buffer);
}

bool iss_true(unsigned value)
{
    if(not (value == TRUE or value == FALSE)){
        throw std::runtime_error("Fault injection attack?");
    }
    return value == TRUE;
}
