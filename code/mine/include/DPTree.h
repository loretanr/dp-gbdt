#ifndef DIFFPRIVTREE_H
#define DIFFPRIVTREE_H

#include "utils.h"
#include "TreeNode.h"
#include <vector>
#include <queue>

using namespace std;

class DPTree
{
private:
    vector<vector<float>> *X;
    vector<float> *y;

    void makeTreeBFS();
    void findBestSplit();

public:
    DPTree(ModelParams *params, DataSet *dataset);
    //: tree_index(tree_index), learning_rate(learning_rate), l2_threshold(l2_threshold), l2_lambda(l2_lambda), privacy_budget(privacy_budget), delta_g(delta_g), delta_v(delta_v), loss(loss), max_depth(max_depth), max_leaves(max_leaves), min_samples_split(min_samples_split), leaf_clipping(leaf_clipping), use_bfs(use_bfs), use_3_trees(use_3_trees), use_decay(use_decay), cat_idx(cat_idx), num_idx(num_idx) {};
    ~DPTree();

    ModelParams *params;
    TreeNode root_node;
    queue<TreeNode *> nodes_bfs;
    vector<TreeNode *> nodes;

    void fit();
};

#endif // DIFFPRIVTREE_H