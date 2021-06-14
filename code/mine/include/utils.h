#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <set>
#include <algorithm>
#include <random>
#include <ctime>
#include <stdexcept>
#include <string>
#include <iostream>
#include <sstream>
#include <numeric>
#include <queue>
#include <cmath>
#include <limits>
#include <iterator>


using namespace std;

struct ModelParams {
    int nb_trees;
    float learning_rate = 0.1;
    float privacy_budget = 1.0;
    //LossFunction loss;
    int max_depth = 6;
    int early_stop = 5;
    int max_leaves;
    int min_samples_split = 2;
    bool second_split = true;
    bool balance_partition = true;
    bool gradient_filtering = false;
    bool leaf_clipping = false;
    bool use_dfs = true;
    bool use_3_trees = false;
    bool use_decay = false;
    int test_size = 0.3;    // TODO 1st or 2nd split?
    int verbosity = -1;
    float l2_threshold = 1.0;
    float l2_lambda = 0.1;
    float delta_g;
    double delta_v;
    float init_score;
    vector<int> cat_idx;
    vector<int> num_idx;
};

struct Scaler {
    float data_min, data_max;
    float feature_min, feature_max;
    Scaler();
    Scaler(float min_val, float max_val, float fmin, float fmax) : data_min(min_val), data_max(max_val),
        feature_min(fmin), feature_max(fmax) {};
};

struct DataSet {
    vector<vector<float>> X;
    vector<float> y;
    vector<float> gradients;
    int length;
    int num_x_cols;
    bool empty;
    Scaler scaler;

    DataSet();
    DataSet(vector<vector<float>> X, vector<float> y);

    void add_row(vector<float> xrow, float yval); // probably broken because theres no self/this
    void scale(float lower, float upper);
};

struct TrainTestSplit {
    DataSet train;
    DataSet test;
    TrainTestSplit(DataSet train, DataSet test) : train(train), test(test) {};
};

struct SplitCandidate {
    int feature_index;
    float split_value;
    float gain;
    SplitCandidate(int f, float s, float g) : feature_index(f), split_value(s), gain(g) {};
};

float clip(float n, float lower, float upper);
vector<string> split_string(const string &s, char delim);           // TODO enable shuffle
TrainTestSplit train_test_split_random(DataSet dataset, float train_ratio = 0.70, bool shuffle = false);

template <typename Iter>
typename std::iterator_traits<Iter>::value_type log_sum_exp(Iter begin, Iter end);

/* #include <cstdint>
typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef uint32_t uint; */

#endif // UTILS_H