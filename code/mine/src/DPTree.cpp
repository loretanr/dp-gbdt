#include "DPTree.h"

DPTree::DPTree(ModelParams *params, DataSet *dataset): params(params) //: tree_index(tree_index), learning_rate(learning_rate), l2_threshold(l2_threshold), l2_lambda(l2_lambda), privacy_budget(privacy_budget), delta_g(delta_g), delta_v(delta_v), loss(loss), max_depth(max_depth), max_leaves(max_leaves), min_samples_split(min_samples_split), leaf_clipping(leaf_clipping), use_bfs(use_bfs), use_3_trees(use_3_trees), use_decay(use_decay), cat_idx(cat_idx), num_idx(num_idx)
{
    // initialize
    X = dataset->X;
    y = dataset->y;
}

DPTree::~DPTree() {}

void DPTree::fit()
{
    // build tree using BFS
    
}

// Recursively build tree, best-leaf-first approach
void DPTree::makeTreeBFS()
{
    // find best split

}

//Find best split of data using the exponential mechanism
void DPTree::findBestSplit()
{
    // iterate over features


}