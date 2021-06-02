#include "dp_ensemble.h"

DPEnsemble::DPEnsemble(ModelParams *parameters)
{
    params = *parameters; // local copy for now
}
    
DPEnsemble::~DPEnsemble() {};

void DPEnsemble::train(DataSet *dataset)
{
    // init gradients

    // prepare privacy budgets

    // train all trees

    trees = {};
}