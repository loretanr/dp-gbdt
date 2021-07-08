#ifndef TREE_H
#define TREE_H
#include "TreeNode.h"

class Tree {
public:
    Tree(Dataset* dataset, ModelParam params);
    ~Tree();
    void grow_tree();
    void prune_tree();
    void inference_tree(float learning_rate=0.1);
    float inference_tree(vector<float>& instance);
    void print_tree();

private:
    void calculate_gh();
    void recursive_prune_tree(TreeNode* node, int& num_leaves_pruned);
    void recursive_print_tree(TreeNode* node);
    int split_node(TreeNode* node);
    int search_best_attr(TreeNode* node, int& best_attr, float& best_split_value,
                         float& max_gain);
    int search_best_split(TreeNode* node, int attribute, int t, int s,
                          float& split_value, float& gain);
    int search_best_numeric_split(TreeNode* node, int attribute, int t, int s,
                                  float& split_value, float& gain);
    int search_best_categoric_split(TreeNode* node, int attribute, int t, int s,
                                    float& split_value, float& gain);
    float calculate_g(float y, float y_pred);
    float calculate_h(float y, float y_pred);
    float calculate_gain(float G_L, float G_R, float H_L, float H_R);
    float calculate_weight(float G, float H);
    TreeNode* root;
    Dataset* dataset;
    ModelParam params;
    vector<float> g, h;
    map<int, map<string, int>> categoric_attr_maps;
    map<int, map<int, string>> categoric_attr_inverse_maps;
    map<string, int> label_maps;
    map<int, string> label_inverse_maps;
};

#endif // TREE_H
