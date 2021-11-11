#ifndef TREENODE_H
#define TREENODE_H


class TreeNode {
public:
    // constructors
    TreeNode(bool _is_leaf): depth(0), split_attr(-1), split_value(-1), split_gain(-1), is_leaf(_is_leaf) {};
    ~TreeNode() {};

    // fields
    TreeNode *left, *right;
    int depth;
    int split_attr;
    double split_value;
    double split_gain;
    bool is_leaf;
    int lhs_size, rhs_size;
    double prediction; // if it's a leaf

    // methods$
    // bool is_leaf();
};


#endif // TREENODE_H