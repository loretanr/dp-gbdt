#ifndef TREENODE_H
#define TREENODE_H

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>

#include "Dataset.h"
using namespace std;

class TreeNode {
public:
    TreeNode();
    void add_instance(int id);
    vector<int> sort_instances_by_attri(Dataset* dataset, int attribute);
    int num_instances();
    int get(int index);
    bool is_leaf();
    void print_node_info();

    TreeNode* left;
    TreeNode* right;
    int depth;
    string loc;
    int split_attr;
    int split_attr_type; // 1 for numeric, 2 for categoric
    float split_value;
    float split_gain;
    float weight;
    vector<int> instances;
};

#endif // TREENODE_H
