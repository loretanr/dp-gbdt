#ifndef UTILS_H
#define UTILS_H

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

#endif // UTILS_H