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
    int min_samples_split = 2;
    bool balance_partition = true;
    bool gradient_filtering = true;
    bool leaf_clipping = true;
    bool scale_y = false;
    bool use_decay = false;
    double l2_threshold = 1.0;
    double l2_lambda = 0.1;
    int verbosity = -1;
    std::vector<int> cat_idx;
    std::vector<int> num_idx;
};

// each tree has these additional parameters
struct TreeParams {
    double delta_g;
    double delta_v;
    double tree_privacy_budget;
};


#endif /* PARAMETERS_H */
