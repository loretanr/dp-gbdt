#include "dp_tree.h"

 //: tree_index(tree_index), learning_rate(learning_rate), l2_threshold(l2_threshold), l2_lambda(l2_lambda), privacy_budget(privacy_budget), delta_g(delta_g), delta_v(delta_v), loss(loss), max_depth(max_depth), max_leaves(max_leaves), min_samples_split(min_samples_split), leaf_clipping(leaf_clipping), use_bfs(use_bfs), use_3_trees(use_3_trees), use_decay(use_decay), cat_idx(cat_idx), num_idx(num_idx)
DPTree::DPTree(ModelParams *params, DataSet *dataset, float privacy_budget): 
    params(params), 
    dataset(dataset),
    privacy_budget(privacy_budget) {}

DPTree::~DPTree() {}

// Fit the tree to the data
void DPTree::fit()
{
    if (params->use_dfs) {
        vector<int> live_samples(dataset->length);
        std::iota(std::begin(live_samples), std::end(live_samples), 0);
        TreeNode root_node = make_tree_DFS(0, live_samples);
        nodes = collect_nodes(root_node);
    } else {
        throw runtime_error("non-DFS not yet implemented.");
    }
    
}

// Recursively build tree, DFS approach, 1st instance returns root node
TreeNode DPTree::make_tree_DFS(int current_depth, vector<int> live_samples)
{
    // max depth reached or not enough samples -> leaf node
    if ( (current_depth == params->max_depth) or
        live_samples.size() < params->min_samples_split) {
            return make_leaf_node(current_depth);
        }

    // find best split

}


TreeNode DPTree::make_leaf_node(int current_depth)
{
    TreeNode leaf = TreeNode();
    leaf.depth = current_depth;
    //leaf.prediction = compute_predictions(nullptr, nullptr);
    nodes.push_back(leaf);
}


vector<TreeNode> DPTree::collect_nodes(TreeNode rootnode)
{

}


float DPTree::compute_predictions(vector<float> gradients, vector<float> y)
{
    return 0.0f;
}


//Find best split of data using the exponential mechanism
void DPTree::find_best_split()
{
    // iterate over features


}