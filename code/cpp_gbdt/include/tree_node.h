#ifndef TREENODE_H
#define TREENODE_H

#include "utils.h"

class TreeNode {
public:
    TreeNode(bool is_leaf);
    ~TreeNode();

    TreeNode *left, *right;
    int depth;
    int split_attr;
    double split_value;
    double split_gain;
    int lhs_size, rhs_size;
    double weight;  // TODO remove?
    double prediction; // if it's a leaf

    bool is_leaf();
};


#endif // TREENODE_H