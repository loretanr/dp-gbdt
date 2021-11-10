#include <numeric>
#include <cmath>
#include <algorithm>
#include "dp_tree.h"
#include "laplace.h"
#include "constant_time.h"

using namespace std;

/** Constructors */

DPTree::DPTree(ModelParams *_params, TreeParams *_tree_params, DataSet *_dataset, size_t _tree_index): 
    params(_params),
    tree_params(_tree_params), 
    dataset(_dataset),
    tree_index(_tree_index) 
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

    this->root_node = make_tree_DFS(0, live_samples);


    // leaf clipping. Note, it can only be disabled if GDF is enabled.
    if (params->leaf_clipping or !params->gradient_filtering) {
        double threshold = params->l2_threshold * std::pow((1 - params->learning_rate), tree_index);
        for (auto &leaf : this->leaves) {
            leaf->prediction = clamp(leaf->prediction, -threshold, threshold);
        }
    }

    // add laplace noise to leaf values
    double privacy_budget_for_leaf_nodes = tree_params->tree_privacy_budget  / 2;
    double laplace_scale = tree_params->delta_v / privacy_budget_for_leaf_nodes;
    add_laplacian_noise(laplace_scale);

}


// Recursively build tree, DFS approach, first instance returns root node
TreeNode *DPTree::make_tree_DFS(int current_depth, vector<int> live_samples)
{
    int live_size = std::accumulate(live_samples.begin(), live_samples.end(), 0);
    bool reached_max_depth = (current_depth == params->max_depth);
    bool not_enough_live_samples = (live_size < params->min_samples_split);
    bool create_leaf_node = constant_time::logical_or(reached_max_depth, not_enough_live_samples);

    // max depth reached or not enough samples -> leaf node
    TreeNode *leaf = make_leaf_node(current_depth, live_samples);

    // find best split
    TreeNode *node = find_best_split(X_transposed, dataset->gradients, live_samples, current_depth, create_leaf_node);

    // no split found or should create leaf node anyways
    if (constant_time::logical_or(node->is_leaf(), create_leaf_node)) {
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
        left_live_samples[i] = lhs[i] * (live_samples[i] == 1);     // number comparisons are constant time
        right_live_samples[i] = rhs[i] * (live_samples[i] == 1);
    }

    node->left = make_tree_DFS(current_depth + 1, left_live_samples);
    node->right = make_tree_DFS(current_depth + 1, right_live_samples);

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
    if(node->is_leaf()){
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
    return constant_time::select(node->is_leaf(), node->prediction, next_level_prediction);
}


// find best split of data using the exponential mechanism
TreeNode *DPTree::find_best_split(VVD &X_transposed, std::vector<double> &gradients_live, std::vector<int> &live_samples, int current_depth, bool create_leaf_node)
{
    double privacy_budget_for_node;
    if (params->use_decay) {
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

        bool categorical = std::find((params->cat_idx).begin(), (params->cat_idx).end(), feature_index) != (params->cat_idx).end();

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

    // choose a split using the exponential mechanism                       // TODO check all other location for const time
    int index = exponential_mechanism(probabilities);                       // TODO return a split candidate here, very inefficient
                                                                            // or is it that bad ? überlege nomol
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

    if(create_leaf_node) {
        int live_size = std::accumulate(live_samples.begin(), live_samples.end(), 0);
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

    // useless split -> return (-1) instead
    total_gain = constant_time::select(useless_split, -1., total_gain);

    return total_gain;
}


// the result is am int array that will indicate left/right resp. 0/1
void DPTree::samples_left_right_partition(vector<int> &lhs, vector<int> &rhs, VVD &samples, vector<int> &live_samples, int feature_index, double feature_value, bool categorical)
{
    // the resulting partition is stored in "lhs/rhs". 
    // - lhs[row]=1 means the row is live and goes to the left child on this split index/value
    // - rhs[row]=1 means the row is live and goes to the right child on this split index/value
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

    // create a cumulative distribution from the probabilities.
    // all values will be in [0,1]
    std::partial_sum(probabilities.begin(), probabilities.end(), partials.begin());

    double rand01 = ((double) sgx_random_pos_int() / (RAND_MAX));

    size_t result_index = 0;
    bool found = false;
    for (size_t index=0; index<partials.size(); index++) {
        result_index = constant_time::select(found, result_index, index);
        found = (partials[index] >= rand01);
    }
    return constant_time::select(no_split_available, -1, (int) result_index);
}


void DPTree::add_laplacian_noise(double laplace_scale)
{
    Laplace lap(laplace_scale, sgx_random_pos_int());

    // add noise from laplace distribution to leaves
    for (auto &leaf : leaves) {
        double noise = lap.return_a_random_variable(laplace_scale);
        leaf->prediction += noise;
    }
}


// free allocated ressources
void DPTree::delete_tree(TreeNode *node)
{
    if (not node->is_leaf()) {
        delete_tree(node->left);
        delete_tree(node->right);
    }
    delete node;
    return;
}