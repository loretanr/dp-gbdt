#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <memory>
#include <vector>
#include "utils.h"
#include "loss.h"


struct ModelParams {
    int nb_trees;
    double learning_rate = 0.1;
    double privacy_budget = 1.0;
    std::shared_ptr<Task> task;
    int max_depth = 6;
    int min_samples_split = 2;
    unsigned balance_partition = TRUE;
    unsigned gradient_filtering = TRUE;
    unsigned leaf_clipping = TRUE;
    unsigned scale_y = FALSE;
    unsigned use_decay = FALSE;
    double l2_threshold = 1.0;
    double l2_lambda = 0.1;
    int verbosity = -1;
    std::vector<int> cat_idx;
    std::vector<int> num_idx;
    unsigned use_grid = FALSE;
    std::tuple<double,double> grid_borders;
    double grid_step_size;
    std::vector<std::vector<double>> cat_values;
    unsigned scale_X = FALSE;
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
