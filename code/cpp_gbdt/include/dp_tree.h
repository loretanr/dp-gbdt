#ifndef DIFFPRIVTREE_H
#define DIFFPRIVTREE_H

#include "utils.h"
#include "tree_node.h"
#include "spdlog/spdlog.h"

extern std::ofstream verification_logfile;


class DPTree
{
private:
    ModelParams *params;
    TreeParams *tree_params;
    DataSet *dataset;
    size_t tree_index;
    vector<set<double>> X_unique;
    vector<TreeNode *> leaves;
    vector<TreeNode> *nodes;  // TODO = necessary?

    TreeNode *make_tree_DFS(int current_depth, vector<int> live_samples);

    TreeNode *make_leaf_node(int current_depth, vector<int> &live_samples);
    double compute_prediction(vector<double> gradients, vector<double> y);
    double _predict(vector<double> *row, TreeNode *node);
    vector<TreeNode> collect_nodes(TreeNode rootnode);

    TreeNode *find_best_split(VVD &X_live, vector<double> &gradients_live, int current_depth);
    void samples_left_right_partition(vector<bool> &lhs, VVD &samples, vector<double> &gradients_live, int feature_index, double feature_value);
    double compute_gain(VVD &samples, vector<double> &gradients_live, int feature_index, double feature_value, int &lhs_size);
    int exponential_mechanism(vector<SplitCandidate> &probs, double max_gain);
    void add_laplacian_noise(double laplace_scale);

public:
    DPTree(ModelParams *params, TreeParams *tree_params, DataSet *dataset, size_t tree_index);
    ~DPTree();

    void delete_tree(TreeNode *node);
    void recursive_print_tree(TreeNode* node);

    TreeNode *root_node;

    vector<double> predict(VVD &X);
    void fit();
};

#endif // DIFFPRIVTREE_H