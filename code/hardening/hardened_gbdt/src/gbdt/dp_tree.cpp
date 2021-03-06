
#include <numeric>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "dp_tree.h"
#include "laplace.h"
#include "logging.h"
#include "constant_time.h"
#include "spdlog/spdlog.h"

extern std::ofstream verification_logfile;
extern bool VERIFICATION_MODE;

using namespace std;


/** Constructors */

DPTree::DPTree(ModelParams *params, TreeParams *tree_params, DataSet *dataset, size_t tree_index, std::vector<double> grid): 
    params(params),
    tree_params(tree_params), 
    dataset(dataset),
    tree_index(tree_index),
    grid(grid)
{
    // only need to transpose X once
    X_transposed = VVD(dataset->num_x_cols, vector<double>(dataset->length));
    for (int row=0; row<dataset->length; row++) {
        for(int col=0; col < dataset->num_x_cols; col++) {
            X_transposed[col][row] = (dataset->X)[row][col];
        }
    }
}


/** Methods */

// Fit the tree to the data
void DPTree::fit()
{
    // keep track which samples will be available in a node for spliting (1=live)
    vector<int> live_samples(dataset->length, 1);

    this->root_node = make_tree_dfs(0, live_samples);

    // leaf clipping. Note, it can only be disabled if GDF is enabled.
    if (is_true(params->leaf_clipping) or !is_true(params->gradient_filtering)) {
        double threshold = params->l2_threshold * std::pow((1 - params->learning_rate), tree_index);
        for (auto &node : this->leaves) {
            node->prediction = clamp(node->prediction, -threshold, threshold);
        }
    }

    // add laplace noise to leaf values
    double privacy_budget_for_leaf_nodes = tree_params->tree_privacy_budget  / 2;
    double laplace_scale = tree_params->delta_v / privacy_budget_for_leaf_nodes;
    add_laplacian_noise(laplace_scale);
}


// Recursively build tree, DFS approach, first instance returns root node
TreeNode *DPTree::make_tree_dfs(int current_depth, vector<int> live_samples)
{
    // int live_size = std::accumulate(live_samples.begin(), live_samples.end(), 0);
    // bool reached_max_depth = (current_depth == params->max_depth);
    // bool not_enough_live_samples = (live_size < params->min_samples_split);
    // // "create_leaf_node" marks whether we would be in the base case of a normal recurstion,
    // // i.e. whether a leaf should be created.
    // bool create_leaf_node = constant_time::logical_or(reached_max_depth, not_enough_live_samples);

    // // find best split
    // TreeNode *node = find_best_split(X_transposed, dataset->gradients, live_samples, current_depth, is_dummy, create_leaf_node);
    // // "node" can be one of three things:
    // // (1) a legitimate leaf node (either we reached max_depth, or a useful split does not exist)
    // // (2) a legitimate internal node
    // // (3) a dummy node (if we already created a legitimate leaf node on this path)

    // bool fake_continuation = constant_time::logical_or(node->is_leaf, is_dummy);
    // // in case (1) and (3) we still need to ensure the continuation of the recursion. For this a random 
    // // feature and feature_value are selected.
    // int random_feature = std::rand() % dataset->num_x_cols ;
    // double random_feature_value = X_transposed[random_feature][ std::rand() % live_samples.size() ];
    // // the following statements carry out the assignment in constant time:
    // // case (1) or (3) -> use the random values
    // // case (2) -> use the real split that was found
    // node->split_attr = constant_time::select(fake_continuation, random_feature, node->split_attr);
    // node->split_value = constant_time::select(fake_continuation, random_feature_value, node->split_value);

    // if(is_dummy) {
    //     LOG_DEBUG("noise recursion, curr_depth {1}", current_depth);
    // } else if (current_depth == params->max_depth) {
    // } else if(fake_continuation) {
    //     LOG_DEBUG("no split found -> leaf");
    // } else {
    //     LOG_DEBUG("best split @ {1}, val {2:.2f}, gain {3:.5f}, depth {4}, samples {5} ->({6},{7})", 
    //         node->split_attr, node->split_value, node->split_gain, current_depth, 
    //         node->lhs_size + node->rhs_size, node->lhs_size, node->rhs_size);
    // }

    // // prepare the new R/L live_samples to continue the recursion, constant time
    // vector<int> lhs(live_samples.size(),0);
    // vector<int> rhs(live_samples.size(),0);

    // samples_left_right_partition(lhs, rhs, X_transposed, live_samples, node->split_attr, node->split_value);

    // vector<int> left_live_samples(live_samples.size(),0);
    // vector<int> right_live_samples(live_samples.size(),0);
    // for (size_t i=0; i<live_samples.size(); i++) {
    //     left_live_samples[i] = lhs[i] * (live_samples[i] == 1);     // number comparisons are constant time
    //     right_live_samples[i] = rhs[i] * (live_samples[i] == 1);
    // }

    // // always recurse until we reach max_depth
    // if(current_depth < params->max_depth){
    //     node->left = make_tree_dfs(current_depth + 1, left_live_samples, 
    //         constant_time::logical_or(is_dummy, create_leaf_node));
    //     node->right = make_tree_dfs(current_depth + 1, right_live_samples,
    //         constant_time::logical_or(is_dummy, create_leaf_node));
    // }

    // return node;
    int live_size = std::accumulate(live_samples.begin(), live_samples.end(), 0);
    // sgx_printf("live_size: %i\n", live_size);
    bool reached_max_depth = (current_depth == params->max_depth);
    bool not_enough_live_samples = (live_size < params->min_samples_split);
    bool create_leaf_node = constant_time::logical_or(reached_max_depth, not_enough_live_samples);

    // max depth reached or not enough samples -> leaf node
    TreeNode *leaf = make_leaf_node(current_depth, live_samples);

    // find best split
    TreeNode *node = find_best_split(X_transposed, dataset->gradients, live_samples, current_depth, create_leaf_node);

    // no split found or should create leaf node anyways
    if (constant_time::logical_or(node->is_leaf, create_leaf_node)) {
        TreeNode *return_node = (TreeNode *) constant_time::select(create_leaf_node, (unsigned long) leaf, (unsigned long) node);
        return return_node;
    }
    bool categorical = std::find((params->cat_idx).begin(), (params->cat_idx).end(), node->split_attr) != (params->cat_idx).end();

    // prepare the new R/L live_samples to continue the recursion, constant time
    vector<int> lhs(live_samples.size(),0);
    vector<int> rhs(live_samples.size(),0);

    samples_left_right_partition(lhs, rhs, X_transposed, live_samples, node->split_attr, node->split_value, categorical);

    vector<int> left_live_samples(live_samples.size(),0);
    vector<int> right_live_samples(live_samples.size(),0);
    for (size_t i=0; i<live_samples.size(); i++) {
        left_live_samples[i] = lhs[i] * (live_samples[i] == 1);
        right_live_samples[i] = rhs[i] * (live_samples[i] == 1);
    }

    node->left = make_tree_dfs(current_depth + 1, left_live_samples);
    node->right = make_tree_dfs(current_depth + 1, right_live_samples);

    return node;
}


TreeNode *DPTree::make_leaf_node(int current_depth, vector<int> &live_samples)
{
    TreeNode *leaf = new TreeNode(true);
    leaf->depth = current_depth;

    // compute prediction
    double gradients_sum = 0;
    int live_size = 0;
    for(size_t index=0; index<live_samples.size(); index++){
        gradients_sum += dataset->gradients[index] * live_samples[index];
        live_size += live_samples[index];
    }
    leaf->prediction = (-1 * gradients_sum / (live_size + params->l2_lambda));
    leaves.push_back(leaf);

    return(leaf);
}


vector<double> DPTree::predict(VVD &X)
{
    vector<double> predictions;
    // iterate over all samples
    for (auto row : X) {
        double pred = _predict(&row, root_node);
        predictions.push_back(pred);
    }

    return predictions;
}


// recursively walk through decision tree
double DPTree::_predict(vector<double> *row, TreeNode *node)
{
    // bool categorical = false;
    // for(auto cat_feature : params->cat_idx){
    //     categorical = constant_time::logical_or(categorical, cat_feature == node->split_attr);
    // }

    // double next_level_prediction = 0.0;

    // // always recurse to max_depth, but not further
    // if(node->depth < params->max_depth){
    //     double row_val = (*row)[node->split_attr];

    //     double left_result = _predict(row, node->left);
    //     double right_result = _predict(row, node->right);
    //     // to hide the real path a sample row takes, we will go down both paths at every
    //     // internal node.
    //     // Further we hide whether the current node splits on a categorical/numerical feature. 
    //     // Which is kinda unnecessary, as the proof gives this
    //     // to the adversary. however it might allow for a tighter proof later.
    //     next_level_prediction = constant_time::select(categorical,
    //             constant_time::select((row_val == node->split_value), left_result, right_result),
    //             constant_time::select((row_val < node->split_value), left_result, right_result) );
    // }

    // // decide whether to take the current node's prediction, or the prediction of its sucessors
    // return constant_time::select(node->is_leaf, node->prediction, next_level_prediction);
    if(node->is_leaf){
        return node->prediction;
    }

    bool categorical = std::find((params->cat_idx).begin(), (params->cat_idx).end(), node->split_attr) != (params->cat_idx).end();

    double next_level_prediction = 0.0;

    if(node->depth < params->max_depth){
        double row_val = (*row)[node->split_attr];

        double left_result = _predict(row, node->left);
        double right_result = _predict(row, node->right);
        // to hide the real path a sample row takes, we will go down both paths at every
        // internal node.
        // Further we hide whether the current node splits on a categorical/numerical feature. 
        // Which is kinda unnecessary, as the proof gives this
        // to the adversary. however it might allow for a tighter proof later.
        next_level_prediction = constant_time::select(categorical,
                constant_time::select((row_val == node->split_value), left_result, right_result),
                constant_time::select((row_val < node->split_value), left_result, right_result) );
    }

    // decide whether to take the current node's prediction, or the prediction of its sucessors
    return constant_time::select(node->is_leaf, node->prediction, next_level_prediction);
}


// find best split of data using the exponential mechanism
TreeNode *DPTree::find_best_split(VVD &X_transposed, vector<double> &gradients_live, vector<int> &live_samples,
    int current_depth, bool create_leaf_node)
{
    double privacy_budget_for_node;
    if (is_true(params->use_decay)) {
        if (current_depth == 0) {
            privacy_budget_for_node = tree_params->tree_privacy_budget /
                (2 * pow(2, params->max_depth + 1) + 2 * pow(2, current_depth + 1));
        } else {
            privacy_budget_for_node = tree_params->tree_privacy_budget / (2 * pow(2, current_depth + 1));
        }
    } else {
        privacy_budget_for_node = (tree_params->tree_privacy_budget) / (2 * params->max_depth );
    }

    vector<SplitCandidate> probabilities;
    int lhs_size;
    
    // iterate over features
    for (int feature_index=0; feature_index < dataset->num_x_cols; feature_index++) {

        bool categorical = std::find( params->cat_idx.begin(), params->cat_idx.end(), feature_index) != params->cat_idx.end();
        
        if(is_true(params->use_grid)){

            if (categorical) {
                for (double feature_value : params->cat_values[feature_index]) {
                    // compute gain
                    double gain = compute_gain(X_transposed, gradients_live, live_samples, feature_index, feature_value, lhs_size, categorical);

                    bool row_not_live = false;
                    bool no_gain = (gain == -1.);
                    // if either the row is not live, or the gain is -1 (aka the split guides all samples to the
                    // same child -> useless split) then the gain of this split is set to 0.
                    gain = constant_time::select(constant_time::logical_or(no_gain, row_not_live), 0.0, gain);

                    // Gi = epsilon_nleaf * Gi / (2 * delta_G)
                    gain = (privacy_budget_for_node * gain) / (2 * tree_params->delta_g);

                    SplitCandidate candidate = SplitCandidate(feature_index, feature_value, gain);
                    candidate.lhs_size = lhs_size;
                    candidate.rhs_size = std::accumulate(live_samples.begin(), live_samples.end(), 0) - lhs_size;
                    probabilities.push_back(candidate); 
                }
            } else {
                for (double feature_value : this->grid) {

                    // double feature_value = X_transposed[feature_index][row];

                    // compute gain
                    double gain = compute_gain(X_transposed, gradients_live, live_samples, feature_index, feature_value, lhs_size, categorical);

                    bool row_not_live = false; //constant_time::logical_not(live_samples[row]);
                    bool no_gain = (gain == -1.);
                    // if either the row is not live, or the gain is -1 (aka the split guides all samples to the
                    // same child -> useless split) then the gain of this split is set to 0.
                    gain = constant_time::select(constant_time::logical_or(no_gain, row_not_live), 0.0, gain);

                    // Gi = epsilon_nleaf * Gi / (2 * delta_G)
                    gain = (privacy_budget_for_node * gain) / (2 * tree_params->delta_g);

                    SplitCandidate candidate = SplitCandidate(feature_index, feature_value, gain);
                    candidate.lhs_size = lhs_size;
                    candidate.rhs_size = std::accumulate(live_samples.begin(), live_samples.end(), 0) - lhs_size;
                    probabilities.push_back(candidate);
                }
            }

        } else { // don't use grid

            for (size_t row=0; row<live_samples.size(); row++) {

                double feature_value = X_transposed[feature_index][row];


                // compute gain
                double gain = compute_gain(X_transposed, gradients_live, live_samples, feature_index, feature_value, lhs_size, categorical);

                bool row_not_live = constant_time::logical_not(live_samples[row]);
                bool no_gain = (gain == -1.);
                // if either the row is not live, or the gain is -1 (aka the split guides all samples to the
                // same child -> useless split) then the gain of this split is set to 0.
                gain = constant_time::select(constant_time::logical_or(no_gain, row_not_live), 0.0, gain);

                // Gi = epsilon_nleaf * Gi / (2 * delta_G)
                gain = (privacy_budget_for_node * gain) / (2 * tree_params->delta_g);

                SplitCandidate candidate = SplitCandidate(feature_index, feature_value, gain);
                candidate.lhs_size = lhs_size;
                candidate.rhs_size = std::accumulate(live_samples.begin(), live_samples.end(), 0) - lhs_size;
                probabilities.push_back(candidate);
            }
        }

    }

    // choose a split using the exponential mechanism
    int index = exponential_mechanism(probabilities);

    // start by constructing a leaf node
    TreeNode *node = make_leaf_node(current_depth, live_samples);

    // if an internal node should be created, change attributes accordingly
    bool create_internal_node = constant_time::logical_and(index != -1, constant_time::logical_not(create_leaf_node));

    // go through all candidates, to hide which one was chosen
    SplitCandidate chosen_one = SplitCandidate(0,0,0);
    chosen_one.lhs_size = 0;
    chosen_one.rhs_size = 0;
    for (int i=0; i<(int)probabilities.size(); i++) {
        bool cond = (i == index);
        chosen_one.feature_index = constant_time::select(cond, probabilities[i].feature_index, chosen_one.feature_index);
        chosen_one.split_value = constant_time::select(cond, probabilities[i].split_value, chosen_one.split_value);
        chosen_one.gain = constant_time::select(cond, probabilities[i].gain, chosen_one.gain);
        chosen_one.lhs_size = constant_time::select(cond, probabilities[i].lhs_size, chosen_one.lhs_size);
        chosen_one.rhs_size = constant_time::select(cond, probabilities[i].rhs_size, chosen_one.rhs_size);
    }
    node->split_attr = constant_time::select(create_internal_node, chosen_one.feature_index, node->split_attr);
    node->split_value = constant_time::select(create_internal_node, chosen_one.split_value, node->split_value);
    node->split_gain = constant_time::select(create_internal_node, chosen_one.gain, node->split_gain);
    node->lhs_size = constant_time::select(create_internal_node, chosen_one.lhs_size, node->lhs_size);
    node->rhs_size = constant_time::select(create_internal_node, chosen_one.rhs_size, node->rhs_size);
    node->is_leaf = constant_time::logical_not(create_internal_node);

    if(create_leaf_node) {
        int live_size = std::accumulate(live_samples.begin(), live_samples.end(), 0);
        LOG_DEBUG("max_depth ({1}) or min_samples ({2})-> leaf (pred={3:.2f})",
            current_depth, live_size, node->prediction);
    }

    return node;
}


/*
    Computes the gain of a split

               sum(elem : IL)^2  +  sum(elem : IR)^2
    G(IL,IR) = ----------------     ----------------
                |IL| + lambda        |IR| + lambda
*/
double DPTree::compute_gain(VVD &X_transposed, vector<double> &gradients_live, vector<int> &live_samples,
    int feature_index, double feature_value, int &lhs_size, bool categorical)
{
    // partition sample rows into lhs / rhs
    vector<int> lhs(live_samples.size(),0);
    vector<int> rhs(live_samples.size(),0);

    samples_left_right_partition(lhs, rhs, X_transposed, live_samples, feature_index, feature_value, categorical);

    int _lhs_size = std::accumulate(lhs.begin(), lhs.end(), 0);
    int _rhs_size = std::accumulate(rhs.begin(), rhs.end(), 0);

    lhs_size = _lhs_size;

    // if all samples go on the same side it's useless to split on this value
    bool useless_split = constant_time::logical_or(_lhs_size == 0, _rhs_size == 0);

    double lhs_gain = 0, rhs_gain = 0;
    for (size_t index=0; index<live_samples.size(); index++) {
        lhs_gain += lhs[index] * (gradients_live)[index];
        rhs_gain += rhs[index] * (gradients_live)[index];
    }
    lhs_gain = std::pow(lhs_gain,2) / (_lhs_size + params->l2_lambda);
    rhs_gain = std::pow(rhs_gain,2) / (_rhs_size + params->l2_lambda);

    double total_gain = lhs_gain + rhs_gain;
    // total_gain = max(total_gain, 0.0);
    total_gain = constant_time::select(total_gain < 0.0, 0.0, total_gain);

    if(VERIFICATION_MODE){
        // round to 10 decimals to avoid numeric issues in verification
        total_gain = std::floor(total_gain * 1e10) / 1e10;
    }

    // useless split -> return (-1) instead
    total_gain = constant_time::select(useless_split, -1., total_gain);

    return total_gain;
}


void DPTree::samples_left_right_partition(vector<int> &lhs, vector<int> &rhs, VVD &samples,
    vector<int> &live_samples, int feature_index, double feature_value, bool categorical)
{
    // // if the feature is categorical, was an unnecessary harden.
    // bool categorical = false;
    // for(auto cat_feature : params->cat_idx){
    //     categorical = constant_time::logical_or(categorical, cat_feature == feature_index);
    // }

    // the resulting partition is stored in "lhs/rhs". 
    // - lhs[row]=1 means the row is live and goes to the left child on this split index/value
    // - rhs[row]=1 means the row is live and goes to the right child on this split index/value
    // for(size_t row=0; row<live_samples.size(); row++){
    //     lhs[row] = constant_time::select(live_samples[row],
    //         (int) constant_time::select(categorical, samples[feature_index][row] == feature_value, samples[feature_index][row] < feature_value), 0);
    //     rhs[row] = constant_time::select(live_samples[row],
    //         (int) constant_time::select(categorical, samples[feature_index][row] != feature_value, samples[feature_index][row] >= feature_value), 0);
    // }
    for(size_t row=0; row<live_samples.size(); row++){
        if(categorical) {
            lhs[row] = constant_time::select(live_samples[row], (int) (samples[feature_index][row] == feature_value), 0);
            rhs[row] = constant_time::select(live_samples[row], (int) (samples[feature_index][row] != feature_value), 0);
        } else {
            lhs[row] = constant_time::select(live_samples[row], (int) (samples[feature_index][row] < feature_value), 0);
            rhs[row] = constant_time::select(live_samples[row], (int) (samples[feature_index][row] >= feature_value), 0);
        }
    }
}


// Computes probabilities from the gains. (Larger gain -> larger probability to 
// be chosen for split). Then a cumulative distribution function is created from
// these probabilities. Then we can sample from it using a RNG.
// The function returns the index of the chosen split.
int DPTree::exponential_mechanism(vector<SplitCandidate> &candidates)
{
    // if no split has a positive gain -> return -1. Node will become a leaf
    int num_viable_candidates = 0;
    for(auto candidate : candidates){
        num_viable_candidates += (candidate.gain > 0);                    
    }
    bool no_split_available = (num_viable_candidates == 0);

    // calculate the probabilities from the gains
    vector<double> gains, probabilities, partials(candidates.size());
    for (auto p : candidates) {
        gains.push_back(p.gain);
    }
    double lse = log_sum_exp(gains);
    for (auto prob : candidates) {
        // let probability be 0 when gain is <= 0
        double probability = constant_time::select(prob.gain <= 0, 0.0, exp(prob.gain - lse));
        probabilities.push_back(probability);   
    }

    if (VERIFICATION_MODE) {
        // non-dp: deterministically choose the best split
        auto max_elem = std::max_element(probabilities.begin(), probabilities.end());
        // return index of the max_elem
        return no_split_available ? -1 : std::distance(probabilities.begin(), max_elem);
    }

    // create a cumulative distribution from the probabilities.
    // all values will be in [0,1]
    std::partial_sum(probabilities.begin(), probabilities.end(), partials.begin());

    double rand01 = ((double) std::rand() / (RAND_MAX));

    size_t result_index = 0;
    bool found = false;
    for (size_t index=0; index<partials.size(); index++) {
        result_index = constant_time::select(found, result_index, index);
        found = (partials[index] >= rand01);
    }
    return constant_time::select(no_split_available, -1, (int) result_index);
}


void DPTree::
add_laplacian_noise(double laplace_scale)
{
    if(VERIFICATION_MODE){
        int num_real_leaves = 0;
        double sum = 0;
        for (auto node : leaves) {
            if(node->is_leaf){
                sum += node->prediction;
                num_real_leaves++;
            }
        }
        sum = sum < 0 && sum >= -1e-10 ? 0 : sum;
        LOG_DEBUG("NUMLEAVES {1} LEAFSUM {2:.8f}", num_real_leaves, sum);
        VERIFICATION_LOG("LEAFVALUESSUM {0:.10f}", sum);
        return;
    }

    LOG_DEBUG("Adding Laplace noise to leaves (Scale {1:.2f})", laplace_scale);

    Laplace lap(laplace_scale, rand());

    // add noise from laplace distribution (to all nodes, for the hardened version)
    for (auto &node : leaves) {
        double noise = lap.return_a_random_variable(laplace_scale);
        node->prediction += noise;
        LOG_DEBUG("({1:.3f} -> {2:.8f})", node->prediction, node->prediction+noise);
    }
}

// free allocated ressources
void DPTree::delete_tree(TreeNode *node)
{
    if (not node->is_leaf) {
        delete_tree(node->left);
        delete_tree(node->right);
    }
    delete node;
    return;
}