#ifndef DPENSEMBLE_H
#define DPENSEMBLE_H

#include <vector>
#include <fstream>
#include "dp_tree.h"
#include "loss.h"

#include "utils.h"


extern std::ofstream verification_logfile;
extern size_t cv_fold_index;

class DPEnsemble
{
public:
    DPEnsemble(ModelParams *params);
    ~DPEnsemble();

    void train(DataSet *dataset);
    std::vector<double> predict(VVD &X);
    std::vector<DPTree> trees;

private:
    ModelParams params;
    void distribute_samples(std::vector<DataSet> *storage_vec, DataSet *train_set);
    double init_score;
};

#endif // DPTREEENSEMBLE_H