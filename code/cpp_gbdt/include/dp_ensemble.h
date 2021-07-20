#ifndef DPENSEMBLE_H
#define DPENSEMBLE_H

#include "utils.h"
#include "dp_tree.h"
#include "loss.h"
#include "spdlog/spdlog.h"

extern std::ofstream verification_logfile;
extern size_t cv_fold_index;

class DPEnsemble
{
public:
    DPEnsemble(ModelParams *params);
    ~DPEnsemble();

    void train(DataSet *dataset);
    vector<double> predict(VVD &X);
    vector<DPTree> trees;

private:
    ModelParams params;
    void distribute_samples(vector<DataSet> *storage_vec, DataSet *train_set);
    double init_score;
};

#endif // DPTREEENSEMBLE_H