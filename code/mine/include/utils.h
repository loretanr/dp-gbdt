#ifndef UTILS_H
#define UTILS_H

#include <vector>
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
    // bool use_decay = false;
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

struct DataSet {
    vector<vector<float>> X;
    vector<float> y;
    vector<float> gradients;
    int length;
    int num_x_cols;
    bool empty;
    DataSet();
    DataSet(vector<vector<float>> X, vector<float> y);
    void add_row(vector<float> xrow, float yval);
    void scale(ModelParams params, float lower, float upper);
};

struct TrainTestSplit {
    DataSet train;
    DataSet test;
    TrainTestSplit(DataSet train, DataSet test) : train(train), test(test) {};
};

float clip(float n, float lower, float upper);
vector<string> split_string(const string &s, char delim);           // TODO enable shuffle
TrainTestSplit train_test_split_random(DataSet dataset, float train_ratio = 0.70, bool shuffle = false);

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