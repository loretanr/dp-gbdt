#include <algorithm>
#include <cmath>
#include "utils.h"


/** Global Variables */

bool VERIFICATION_MODE;
bool RANDOMIZATION;
size_t cv_fold_index;


// create some default parameters for quick testing
ModelParams create_default_params()
{
    ModelParams params;
    params.nb_trees = 50;
    params.max_depth = 6;
    params.gradient_filtering = true;
    params.leaf_clipping = true;
    params.privacy_budget = 0.1;
    return params;
};





double clip(double n, double lower, double upper)
{
  return std::max(lower, std::min(n, upper));
}


std::string string_pad(std::string str, const size_t num, const char paddingChar)
{
    if(num > str.size())
        str.insert(0, num - str.size(), paddingChar);
    return str;
}


// TODO does this overflow?
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

