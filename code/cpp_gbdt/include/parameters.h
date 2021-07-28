#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <memory>
#include <vector>
#include "loss.h"


struct ModelParams {
    int nb_trees;
    double learning_rate = 0.1;
    double privacy_budget = 1.0;
    std::shared_ptr<Task> task;
    int max_depth = 6;
    int early_stop = 5;
    int max_leaves;
    int min_samples_split = 2;
    bool second_split = false;
    bool balance_partition = true;
    bool gradient_filtering = false;
    bool leaf_clipping = false;
    bool use_dp = false; // TODO remove this default
    bool use_dfs = true;
    bool use_3_trees = false;
    bool use_decay = false;
    int test_size = 0.3;    // TODO 1st or 2nd split?
    double l2_threshold = 1.0;
    double l2_lambda = 0.1;
    double init_score;
    int verbosity = -1;
    std::vector<int> cat_idx;
    std::vector<int> num_idx;
};

// each tree has these additional varying parameters
struct TreeParams {
    double delta_g;
    double delta_v;
    double tree_privacy_budget;
};


#endif /* PARAMETERS_H */