#include "tree_node.h"


TreeNode::TreeNode(bool _is_leaf): depth(0), split_attr(-1), split_value(-1), split_gain(-1)
{
    if (_is_leaf) {
        left = nullptr; right = nullptr;
        is_leaf = true;
    } else {
        left = this; right = this;
        is_leaf = false;
    }
}

TreeNode::~TreeNode() {};



