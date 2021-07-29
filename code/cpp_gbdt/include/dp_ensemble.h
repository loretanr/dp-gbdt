#ifndef DPENSEMBLE_H
#define DPENSEMBLE_H

#include <vector>
#include <fstream>
#include "dp_tree.h"
#include "parameters.h"
#include "data.h"


class DPEnsemble
{
public:
    // constructors
    DPEnsemble(ModelParams *params);
    ~DPEnsemble();

    // fields
    std::vector<DPTree> trees;

    // methods
    void train(DataSet *dataset);
    std::vector<double> predict(VVD &X);

private:
    // fields
    ModelParams *params;
    DataSet *dataset;
    double init_score;

    // methods
    void distribute_samples(std::vector<DataSet> *storage_vec, DataSet *train_set);
    std::vector<double> update_gradients(std::vector<DataSet> &tree_samples, int tree_index);
};

#endif // DPTREEENSEMBLE_H