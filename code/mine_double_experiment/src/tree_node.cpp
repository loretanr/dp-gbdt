#include "tree_node.h"

TreeNode::TreeNode(bool is_leaf): depth(0), split_attr(-1), split_value(-1), split_gain(-1)
{
    if (is_leaf) {
        left = nullptr; right = nullptr;
    } else {
        left = this; right = this;
    }
}

TreeNode::~TreeNode() {}

bool TreeNode::is_leaf() {
    return (left == nullptr && right == nullptr);
}



