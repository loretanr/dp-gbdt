#ifndef TREENODE_H
#define TREENODE_H


class TreeNode {
public:
    // constructors
    TreeNode(bool _is_leaf);
    ~TreeNode();

    // fields
    TreeNode *left, *right;
    int depth;
    int split_attr;
    double split_value;
    double split_gain;
    int lhs_size, rhs_size;
    bool is_leaf;
    bool is_dummy;
    double prediction; // if it's a leaf
};


#endif // TREENODE_H