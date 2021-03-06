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
    VVD X_transposed; // for hardened gbdt, we always work on the full X, so we only transpose it once
    size_t tree_index;
    std::vector<TreeNode *> nodes;
    std::vector<double> grid;


    // methods
    TreeNode *make_tree_dfs(int current_depth, std::vector<int> live_samples, bool is_dummy);
    TreeNode *make_leaf_node(int current_depth, std::vector<int> &live_samples);
    double _predict(std::vector<double> *row, TreeNode *node);
    TreeNode *find_best_split(VVD &X_transposed, std::vector<double> &gradients_live, std::vector<int> &live_samples, int current_depth, bool is_dummy, bool create_leaf_node);
    void samples_left_right_partition(std::vector<int> &lhs, std::vector<int> &rhs, VVD &samples,  std::vector<int> &live_samples,
                int feature_index, double feature_value);
    double compute_gain(VVD &X_transposed, std::vector<double> &gradients_live, std::vector<int> &live_samples, int feature_index,
                double feature_value, int &lhs_size);
    int exponential_mechanism(std::vector<SplitCandidate> &candidates);
    void add_laplacian_noise(double laplace_scale);

public:
    // constructors
    DPTree(ModelParams *params, TreeParams *tree_params, DataSet *dataset, size_t tree_index, std::vector<double> grid);
    ~DPTree() {};

    // fields
    TreeNode *root_node;

    // methods
    std::vector<double> predict(VVD &X);
    void fit();
    void delete_tree();
};

#endif // DIFFPRIVTREE_H