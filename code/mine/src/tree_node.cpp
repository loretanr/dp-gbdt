#include "tree_node.h"

TreeNode::TreeNode(): left(nullptr), right(nullptr), depth(0),
                      split_attr(-1), split_value(0), split_gain(0) {
}

TreeNode::~TreeNode() {}

bool TreeNode::is_leaf()
{
    return left == nullptr && right == nullptr;
}