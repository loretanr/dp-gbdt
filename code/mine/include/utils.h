#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <algorithm>
#include <random>
#include <ctime>

using namespace std;

struct ModelParams {
    int tree_index;
    float learning_rate;
    float l2_threshold;
    float l2_lambda;
    float privacy_budget;
    float delta_g;
    float delta_v;
    //LossFunction loss;
    int max_depth = 6;
    int max_leaves;
    int min_samples_split = 2;
    bool leaf_clipping = false;
    bool use_bfs = false;
    bool use_3_trees = false;
    bool use_decay = false;
    int *cat_idx;
    int *num_idx;
};

struct DataSet {
    vector<vector<float>> X;
    vector<float> y;
    DataSet(vector<vector<float>> X, vector<float> y) : X(X), y(y) {};
};

struct TrainTestSplit {
    DataSet train;
    DataSet test;
    TrainTestSplit(DataSet train, DataSet test) : train(train), test(test) {};
};

TrainTestSplit train_test_split_random(DataSet dataset, float train_ratio = 0.75);

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