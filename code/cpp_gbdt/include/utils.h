#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <random>
#include <ctime>
#include <stdexcept>
#include <string>
#include <iostream>
#include <iomanip> 
#include <sstream>
#include <numeric>
#include <queue>
#include <cmath>
#include <limits>
#include <iterator>
#include <fstream>
#include <cstdlib>
#include <spdlog/sinks/stdout_sinks.h>
// #include "spdlog/spdlog.h"
#include "loss.h"
#include <memory>


/* Logging, to be removed */
#define concat(one, two) ((std::string) one + (std::string) two).c_str()
#define LOG_INFO_ARG(msg, ...) spdlog::info(concat("[{0:>17s}] ", msg), __func__,  __VA_ARGS__)
#define LOG_INFO_NO_ARG(msg) spdlog::info(concat("[{0:>17s}] ", msg), __func__)
#define LOG_DEBUG_ARG(msg, ...) spdlog::debug(concat("[{0:>17s}] ", msg), __func__,  __VA_ARGS__)
#define LOG_DEBUG_NO_ARG(msg) spdlog::debug(concat("[{0:>17s}] ", msg), __func__)
#define GET_9TH_ARG(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, ...) arg9
#define LOG_INFO_MACRO_CHOOSER(...) \
    GET_9TH_ARG(__VA_ARGS__, LOG_INFO_ARG, LOG_INFO_ARG, LOG_INFO_ARG, LOG_INFO_ARG, LOG_INFO_ARG, LOG_INFO_ARG, LOG_INFO_ARG, LOG_INFO_NO_ARG, )
#define LOG_INFO(...) LOG_INFO_MACRO_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
#define LOG_DEBUG_MACRO_CHOOSER(...) \
    GET_9TH_ARG(__VA_ARGS__, LOG_DEBUG_ARG, LOG_DEBUG_ARG, LOG_DEBUG_ARG, LOG_DEBUG_ARG, LOG_DEBUG_ARG, LOG_DEBUG_ARG, LOG_DEBUG_ARG, LOG_DEBUG_NO_ARG, )
#define LOG_DEBUG(...) LOG_DEBUG_MACRO_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
#define BOLD(words) "\033[0;40;33m" + words + "\033[0m"

#define VERIFICATION_LOG(...) verification_logfile << fmt::format(__VA_ARGS__) << "\n"; verification_logfile.flush()


typedef std::vector<std::vector<double>> VVD;

extern bool RANDOMIZATION;
extern bool VERIFICATION_MODE;
extern size_t cv_fold_index;


using namespace std;

struct ModelParams {
    int nb_trees;
    double learning_rate = 0.1;
    double privacy_budget = 1.0;
    shared_ptr<LossFunction> lossfunction;
    int max_depth = 6;
    int early_stop = 5;
    int max_leaves;
    int min_samples_split = 2;
    bool second_split = true;
    bool balance_partition = true;
    bool gradient_filtering = false;
    bool leaf_clipping = false;
    bool use_dp = false; // TODO remove this default
    bool use_dfs = true;
    bool use_3_trees = false;
    bool use_decay = false;
    int test_size = 0.3;    // TODO 1st or 2nd split?
    int verbosity = -1;
    double l2_threshold = 1.0;
    double l2_lambda = 0.1;

    double init_score;
    vector<int> cat_idx;
    vector<int> num_idx;
};


struct Scaler {
    double data_min, data_max;
    double feature_min, feature_max;
    double scale, min_;
    Scaler() {};
    Scaler(double min_val, double max_val, double fmin, double fmax);
};

struct DataSet {
    VVD X;
    vector<double> y;
    vector<double> gradients;
    int length;
    int num_x_cols;
    bool empty;
    Scaler scaler;
    std::string name;
    std::string task;

    DataSet();
    DataSet(VVD X, vector<double> y);

    void add_row(vector<double> xrow, double yval); // probably broken because theres no self/this
    void scale(double lower, double upper);
};

struct TrainTestSplit {
    DataSet train;
    DataSet test;
    TrainTestSplit(DataSet train, DataSet test) : train(train), test(test) {};
};

struct SplitCandidate {
    int feature_index;
    double split_value;
    double gain;
    SplitCandidate(int f, double s, double g) : feature_index(f), split_value(s), gain(g) {};
};

struct TreeParams {
    double delta_g;
    double delta_v;
    double tree_privacy_budget;
};

ModelParams create_default_params();
void inverse_scale(Scaler &scaler, vector<double> &vec);
double clip(double n, double lower, double upper);
vector<string> split_string(const string &s, char delim);           // TODO enable shuffle
TrainTestSplit train_test_split_random(DataSet dataset, double train_ratio = 0.70, bool shuffle = false);
vector<TrainTestSplit> create_cross_validation_inputs(DataSet &dataset, int folds, bool shuffle);


double log_sum_exp(vector<double> arr);
void string_pad(std::string &str, const size_t num, const char paddingChar = ' ');

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
    Laplace(double scale, int seed): scale(scale), generator(seed), distribution(1.0/scale){};
    double return_a_random_variable();
    double return_a_random_variable(double scale);
};

#endif // UTILS_H