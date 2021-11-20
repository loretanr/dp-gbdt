#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <memory>
#include <vector>
#include "loss.h"


struct ModelParams {
    int nb_trees;
    double learning_rate = 0.1;
    double privacy_budget = 1;
    std::shared_ptr<Task> task;
    int max_depth = 6;
    int min_samples_split = 2;
    bool balance_partition = true;
    bool gradient_filtering = true;
    bool leaf_clipping = true;
    bool use_dp = true;
    bool scale_y = false;
    bool use_decay = false;
    double l2_threshold = 1.0;
    double l2_lambda = 0.1;
    std::vector<int> cat_idx;
    std::vector<int> num_idx;

    // these are all for grid usage
    bool use_grid = false;
    std::tuple<double,double> grid_borders;
    double grid_step_size;
    std::vector<std::vector<double>> cat_values;
    bool scale_X = false;
    double scale_X_percentile = 95;
    double scale_X_privacy_budget = 0.4;
};

// each tree has these additional parameters
struct TreeParams {
    double delta_g;
    double delta_v;
    double tree_privacy_budget;
};


#endif /* PARAMETERS_H */
