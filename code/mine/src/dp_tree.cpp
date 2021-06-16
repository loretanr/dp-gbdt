#include "dp_tree.h"
#include "spdlog/spdlog.h"

//#include <quadmath.h>

 //: tree_index(tree_index), learning_rate(learning_rate), l2_threshold(l2_threshold), l2_lambda(l2_lambda), privacy_budget(privacy_budget), delta_g(delta_g), delta_v(delta_v), loss(loss), max_depth(max_depth), max_leaves(max_leaves), min_samples_split(min_samples_split), leaf_clipping(leaf_clipping), use_bfs(use_bfs), use_3_trees(use_3_trees), use_decay(use_decay), cat_idx(cat_idx), num_idx(num_idx)
DPTree::DPTree(ModelParams *params, TreeParams *tree_params, DataSet *dataset): 
    params(params),
    tree_params(tree_params), 
    dataset(dataset)
{
   /*  // create a matrix whose rows contain the columns of X, but without duplicates
    for (int i=0; i<dataset->num_x_cols; i++){
        X_unique.push_back(set<float>());
    }
    for (int row=0; row < dataset->length; row++) {
        for (int col=0; col < dataset->num_x_cols; col++) {
            X_unique[col].insert(dataset->X[row][col]);
        }
    } */
}

// TODO destroy the nodes, done in ensemble for now.
DPTree::~DPTree() {}

// Fit the tree to the data
void DPTree::fit()
{
    if (params->use_dfs) {
        // keep track which samples will be available in a node for spliting
        vector<int> live_samples(dataset->length);
        std::iota(std::begin(live_samples), std::end(live_samples), 0);

        TreeNode *root_node = make_tree_DFS(0, live_samples);
        nodes = collect_nodes(*root_node);
    } else {
        throw runtime_error("non-DFS not yet implemented.");
    }
    
}

// Recursively build tree, DFS approach, 1st instance returns root node
TreeNode *DPTree::make_tree_DFS(int current_depth, vector<int> live_samples)
{
    // max depth reached or not enough samples -> leaf node
    if ( (current_depth == params->max_depth) or
        live_samples.size() < (size_t) params->min_samples_split) {
            return make_leaf_node(current_depth);
        }

    // find best split
    TreeNode *node = find_best_split(live_samples, current_depth);

    // no split found
    if (node->is_leaf()) {
        return node;
    }

    vector<bool> lhs;
    samples_left_right_partition(lhs, node->split_attr, node->split_value);
    vector<int> left_live_samples, right_live_samples;
    for (auto sample_index : live_samples) {
        if (lhs[sample_index]) {
            left_live_samples.push_back(sample_index);
        } else {
            right_live_samples.push_back(sample_index);
        }
    }

    node->left = make_tree_DFS(current_depth + 1, left_live_samples);
    node->right = make_tree_DFS(current_depth + 1, right_live_samples);



}


TreeNode *DPTree::make_leaf_node(int current_depth)
{
    TreeNode *leaf = new TreeNode();
    leaf->depth = current_depth;
    //leaf.prediction = compute_predictions(nullptr, nullptr);
    //nodes.push_back(leaf); // push back here???
    return(leaf);
}


vector<TreeNode> DPTree::collect_nodes(TreeNode rootnode)
{
    // dummy return val
    vector<TreeNode> bla;
    return bla;
}


float DPTree::compute_predictions(vector<float> gradients, vector<float> y)
{
    return 0.0f;
}


//Find best split of data using the exponential mechanism
TreeNode *DPTree::find_best_split(vector<int> live_samples, int current_depth)
{
    float privacy_budget_for_node;
    if ((current_depth != 0) and params->use_decay) {
        privacy_budget_for_node = (tree_params->tree_privacy_budget) / 2 / pow(2, current_depth);
    } else {
        privacy_budget_for_node = (tree_params->tree_privacy_budget)/2/params->max_depth;
    }
    if (params->use_3_trees and (current_depth != 0)) {
        // Except for the root node, budget is divided by the 3-nodes
        privacy_budget_for_node /= 2;
    }

    LOG_DEBUG("Using {1:.2f} budget for internal leaf nodes", privacy_budget_for_node);

    vector<SplitCandidate> probabilities;
    float max_gain = numeric_limits<float>::min();
    
    // only use the samples that actually end up in this node
    // note that the cols of X are rows in X_live
    vector<vector<float>> X_live;
    for(int col=0; col < dataset->num_x_cols; col++) {
        vector<float> temp;
        for (auto elem : live_samples) {
            temp.push_back((dataset->X)[elem][col]);
        }
        X_live.push_back(temp);
    }

    // iterate over features
    for (int feature_index=0; feature_index < dataset->num_x_cols; feature_index++) {
        for (float feature_value : X_live[feature_index]) { // TODO, don't iterate over duplicates in X_live
            // compute gain
            float gain = compute_gain(feature_index, feature_value);
            // feature cannot be chosen, skipping
            if (gain == -1) {
                continue;
            }
            gain = (privacy_budget_for_node * gain) / (2 * tree_params->delta_g);
            max_gain = (gain > max_gain) ? gain : max_gain; // unused
            SplitCandidate candidate = SplitCandidate(feature_index, feature_value, gain);
            probabilities.push_back(candidate);
        }
    }
    int index = exponential_mechanism(probabilities, max_gain);
    TreeNode *node = new TreeNode();
    if (index == -1) {
        node->left = nullptr;
        node->right = nullptr;
    }
    node->split_attr = probabilities[index].feature_index;
    node->split_value = probabilities[index].split_value;
    node->split_gain = probabilities[index].gain;
    return node;
}


float DPTree::compute_gain(int feature_index, float feature_value)
{
    // partition into lhs / rhs
    vector<bool> lhs;
    samples_left_right_partition(lhs, feature_index, feature_value);

    int lhs_size = std::count(lhs.begin(), lhs.end(), true);
    int rhs_size = std::count(lhs.begin(), lhs.end(), false);

    // if all samples go on the same side it's useless to split on this value
    if ( lhs_size == 0 or rhs_size == 0 ) {
        return -1;
    }

    float lhs_gain = 0, rhs_gain = 0;
    for (size_t index=0; index<lhs.size(); index++) {
        lhs_gain += (unsigned) lhs[index] * (dataset->gradients)[index];
        rhs_gain += (unsigned) (not lhs[index]) * (dataset->gradients)[index];
    }
    lhs_gain = std::pow(lhs_gain,2) / (lhs_size + params->l2_lambda);
    rhs_gain = std::pow(rhs_gain,2) / (rhs_size + params->l2_lambda);
    return std::max(lhs_gain + rhs_gain, 0.0f);
}


void DPTree::samples_left_right_partition(vector<bool> &lhs, int feature_index, float feature_value)
{
    // if the feature is categorical
    if(std::find((params->cat_idx).begin(), (params->cat_idx).end(), feature_index) != (params->cat_idx).end()) {
        for (auto sample : dataset->X) {
            lhs.push_back(sample[feature_index] == feature_value);
        }
    } else { // feature is continuous
        for (auto sample : dataset->X) {
            lhs.push_back(sample[feature_index] < feature_value);
        }
    }
}


int DPTree::exponential_mechanism(vector<SplitCandidate> &probs, float max_gain)
{
    int count = std::count_if(probs.begin(), probs.end(),[](SplitCandidate c){ return c.gain > 0; });

    if (count == 0) {
        return -1;
    }

    // TODO need quadmath for overflows?

    vector<double> gains, probabilities, partials(probs.size());

    for (auto p : probs) {
        if (p.gain != 0) {
            gains.push_back(p.gain);
        }
    }
    for (auto &prob : probs) {
        if (prob.gain <= 0) {
            probabilities.push_back(0);   
        } else {
            probabilities.push_back( exp( (double) prob.gain - log_sum_exp(gains)) );
        }
    }

    // all values in [0,1]
    std::partial_sum(probabilities.begin(), probabilities.end(), partials.begin());

    double rand01 = ((double) rand() / (RAND_MAX));

    // try to find a candidate at least 10 times before giving up and making the node a leaf node
    for (int tries=0; tries<10; tries++) {
        for (size_t index=0; index<partials.size(); index++) {
            if (partials[index] >= rand01) {
                return index;
            }
        }
        rand01 = ((double) rand() / (RAND_MAX));
    }
    return -1;
}

void DPTree::delete_tree(TreeNode *node)
{
    if (not node->is_leaf()) {
        delete_tree(node->left);
        delete_tree(node->right);
    }
    delete node;
    return;
}
