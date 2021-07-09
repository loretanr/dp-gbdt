#ifndef DIFFPRIVTREE_H
#define DIFFPRIVTREE_H

#include "utils.h"
#include "tree_node.h"
#include "spdlog/spdlog.h"


extern std::ofstream validation_logfile;

class DPTree
{
private:
    ModelParams *params;
    TreeParams *tree_params;
    DataSet *dataset;
    vector<set<double>> X_unique;
    vector<TreeNode *> leaves;
    vector<TreeNode> *nodes;  // = necessary?

    TreeNode *make_tree_DFS(int current_depth, vector<int> live_samples);

    TreeNode *make_leaf_node(int current_depth, vector<int> &live_samples);
    double compute_prediction(vector<double> gradients, vector<double> y);
    double _predict(vector<double> *row, TreeNode *node);
    vector<TreeNode> collect_nodes(TreeNode rootnode);

    TreeNode *find_best_split(VVF &X_live, vector<double> &gradients_live, int current_depth);
    void samples_left_right_partition(vector<bool> &lhs, VVF &samples, vector<double> &gradients_live, int feature_index, double feature_value);
    double compute_gain(VVF &samples, vector<double> &gradients_live, int feature_index, double feature_value);
    int exponential_mechanism(vector<SplitCandidate> &probs, double max_gain);
    void add_laplacian_noise(vector<TreeNode *> leaves, double laplace_scale);

public:
    DPTree(ModelParams *params, TreeParams *tree_params, DataSet *dataset);
    //: tree_index(tree_index), learning_rate(learning_rate), l2_threshold(l2_threshold), l2_lambda(l2_lambda), privacy_budget(privacy_budget), delta_g(delta_g), delta_v(delta_v), loss(loss), max_depth(max_depth), max_leaves(max_leaves), min_samples_split(min_samples_split), leaf_clipping(leaf_clipping), use_bfs(use_bfs), use_3_trees(use_3_trees), use_decay(use_decay), cat_idx(cat_idx), num_idx(num_idx) {};
    ~DPTree();

    void delete_tree(TreeNode *node);
    void recursive_print_tree(TreeNode* node);

    TreeNode *root_node;
    //queue<TreeNode *> nodes_bfs;

    vector<double> predict(VVF &X);
    void fit();
};

#endif // DIFFPRIVTREE_H