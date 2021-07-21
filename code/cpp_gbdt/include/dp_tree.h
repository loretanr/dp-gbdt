#ifndef DIFFPRIVTREE_H
#define DIFFPRIVTREE_H

#include <vector>
#include <set>
#include <fstream>
#include "tree_node.h"

#include "utils.h"

extern std::ofstream verification_logfile;


class DPTree
{
private:
    ModelParams *params;
    TreeParams *tree_params;
    DataSet *dataset;
    size_t tree_index;
    std::vector<std::set<double>> X_unique;
    std::vector<TreeNode *> leaves;

    TreeNode *make_tree_DFS(int current_depth, std::vector<int> live_samples);

    TreeNode *make_leaf_node(int current_depth, std::vector<int> &live_samples);
    double compute_prediction(std::vector<double> gradients, std::vector<double> y);
    double _predict(std::vector<double> *row, TreeNode *node);
    std::vector<TreeNode> collect_nodes(TreeNode rootnode);

    TreeNode *find_best_split(VVD &X_live, std::vector<double> &gradients_live, int current_depth);
    void samples_left_right_partition(std::vector<bool> &lhs, VVD &samples, std::vector<double> &gradients_live, int feature_index, double feature_value);
    double compute_gain(VVD &samples, std::vector<double> &gradients_live, int feature_index, double feature_value, int &lhs_size);
    int exponential_mechanism(std::vector<SplitCandidate> &probs, double max_gain);
    void add_laplacian_noise(double laplace_scale);

public:
    DPTree(ModelParams *params, TreeParams *tree_params, DataSet *dataset, size_t tree_index);
    ~DPTree();

    void delete_tree(TreeNode *node);
    void recursive_print_tree(TreeNode* node);

    TreeNode *root_node;

    std::vector<double> predict(VVD &X);
    void fit();
};

#endif // DIFFPRIVTREE_H