#ifndef DIFFPRIVTREE_H
#define DIFFPRIVTREE_H

#include <vector>
#include <set>
#include <fstream>
#include "utils.h"
#include "tree_node.h"
#include "parameters.h"
#include "data.h"


// wrapper around attributes that represent one possible split
struct SplitCandidate {
    int feature_index;
    double split_value;
    double gain;
    int lhs_size, rhs_size;
    SplitCandidate(int f, double s, double g) : feature_index(f), split_value(s), gain(g) {};
};


class DPTree
{
private:
    // fields
    ModelParams *params;
    TreeParams *tree_params;
    DataSet *dataset;
    size_t tree_index;
    std::vector<TreeNode *> leaves;

    // methods
    TreeNode *make_tree_dfs(int current_depth, std::vector<int> live_samples);
    TreeNode *make_leaf_node(int current_depth, std::vector<int> &live_samples);
    double _predict(std::vector<double> *row, TreeNode *node);
    TreeNode *find_best_split(VVD &X_live, std::vector<double> &gradients_live, std::vector<int> &live_samples, int current_depth);
    void samples_left_right_partition(std::vector<int> &lhs, VVD &samples,  std::vector<int> &live_samples,
                int feature_index, double feature_value);
    double compute_gain(VVD &samples, std::vector<double> &gradients_live, std::vector<int> &live_samples, int feature_index,
                double feature_value, int &lhs_size);
    int exponential_mechanism(std::vector<SplitCandidate> &probs);
    void add_laplacian_noise(double laplace_scale);

public:
    // constructors
    DPTree(ModelParams *params, TreeParams *tree_params, DataSet *dataset, size_t tree_index);
    ~DPTree();

    // fields
    TreeNode *root_node;

    // methods
    std::vector<double> predict(VVD &X);
    void fit();
    void recursive_print_tree(TreeNode* node);
    void delete_tree(TreeNode *node);
};

#endif // DIFFPRIVTREE_H