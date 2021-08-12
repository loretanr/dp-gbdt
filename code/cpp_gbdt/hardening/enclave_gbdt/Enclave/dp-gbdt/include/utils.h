#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
typedef std::vector<std::vector<double>> VVD;
#include "parameters.h"
#include "data.h"


// method declarations
ModelParams create_default_params();
double clamp(double n, double lower, double upper);
double log_sum_exp(std::vector<double> arr);
void string_pad(std::string &str, const size_t num, const char paddingChar = ' ');
double compute_mean(std::vector<double> &vec);
double compute_stdev(std::vector<double> &vec, double mean);
std::string get_time_string();

// sgx methods
int sgx_random_int();
template <typename T, typename A>
void sgx_vector_shuffle(std::vector<T,A> &vec)
{
    // Fisher-Yates shuffle, vector of n elements (indices 0..n-1):
    // for i from 0 to n−2 do
    //      j ← random integer such that i ≤ j < n
    //      exchange a[i] and a[j]
    int n = vec.size();
    for (int i = 0; i < n - 1; i++)
    {
        int j = i + sgx_random_int() % (n - i);
        std::swap(vec[i], vec[j]);
    }
}


#endif // UTILS_H