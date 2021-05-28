#ifndef TREENODE_H
#define TREENODE_H

class TreeNode {
public:
    TreeNode();
    ~TreeNode();

    // probably need some info/pointers to dataset

    bool is_leaf();
    //void print_node_info();

    TreeNode *left;
    TreeNode *right;
    int depth;
    int split_index;
    float split_value;
    float split_gain;
    float weight;
    float prediction; // if it's a leaf
};


#endif // TREENODE_H