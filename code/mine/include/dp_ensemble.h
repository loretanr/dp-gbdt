#ifndef DPENSEMBLE_H
#define DPENSEMBLE_H

#include "dp_tree.h"
#include "utils.h"


class DPEnsemble
{
public:
    DPEnsemble(ModelParams *params);
    //: tree_index(tree_index), learning_rate(learning_rate), l2_threshold(l2_threshold), l2_lambda(l2_lambda), privacy_budget(privacy_budget), delta_g(delta_g), delta_v(delta_v), loss(loss), max_depth(max_depth), max_leaves(max_leaves), min_samples_split(min_samples_split), leaf_clipping(leaf_clipping), use_bfs(use_bfs), use_3_trees(use_3_trees), use_decay(use_decay), cat_idx(cat_idx), num_idx(num_idx) {};
    ~DPEnsemble();

    void train(DataSet *dataset);
    vector<float> predict(vector<vector<float>> *X);

private:
    ModelParams params;
    vector<DPTree> trees;
    void distribute_samples(vector<DataSet> *storage_vec, DataSet *train_set);
};

#endif // DPTREEENSEMBLE_H