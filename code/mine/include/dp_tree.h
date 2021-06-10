#ifndef DIFFPRIVTREE_H
#define DIFFPRIVTREE_H

#include "utils.h"
#include "tree_node.h"

using namespace std;

class DPTree
{
private:
    ModelParams *params;
    DataSet *dataset;
    float privacy_budget;

    TreeNode make_tree_DFS(int current_depth, vector<int> live_samples);

    TreeNode make_leaf_node(int current_depth);
    float compute_predictions(vector<float> gradients, vector<float> y);
    vector<TreeNode> collect_nodes(TreeNode rootnode);

    void find_best_split();

public:
    DPTree(ModelParams *params, DataSet *dataset, float privacy_budget);
    //: tree_index(tree_index), learning_rate(learning_rate), l2_threshold(l2_threshold), l2_lambda(l2_lambda), privacy_budget(privacy_budget), delta_g(delta_g), delta_v(delta_v), loss(loss), max_depth(max_depth), max_leaves(max_leaves), min_samples_split(min_samples_split), leaf_clipping(leaf_clipping), use_bfs(use_bfs), use_3_trees(use_3_trees), use_decay(use_decay), cat_idx(cat_idx), num_idx(num_idx) {};
    ~DPTree();

    TreeNode root_node;
    //queue<TreeNode *> nodes_bfs;
    vector<TreeNode> nodes;  // main list of nodes

    void fit();
};

#endif // DIFFPRIVTREE_H