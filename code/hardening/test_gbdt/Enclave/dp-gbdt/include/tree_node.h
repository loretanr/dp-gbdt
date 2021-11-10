#ifndef TREENODE_H
#define TREENODE_H


class TreeNode {
public:
    // constructors
    TreeNode(bool is_leaf);
    ~TreeNode();

    // fields
    TreeNode *left, *right;
    int depth;
    int split_attr;
    double split_value;
    double split_gain;
    int lhs_size, rhs_size;
    double prediction; // if it's a leaf

    // methods
    bool is_leaf();
};


#endif // TREENODE_H