#include "dp_ensemble.h"

DPEnsemble::DPEnsemble(ModelParams *parameters)
{
    params = *parameters; // local copy for now
}
    
DPEnsemble::~DPEnsemble() {};

void DPEnsemble::train(DataSet *dataset)
{
    // update gradients

    // second split
    float prev_score = numeric_limits<float>::max();
    TrainTestSplit split = train_test_split_random(*dataset);
    DataSet *train_set = &split.train;
    DataSet *test_set = &split.test;

    // prepare privacy budgets
    float tree_privacy_budget = params.privacy_budget / params.nb_trees;

    // train all trees
    for(int tree_index = 0; tree_index<params.nb_trees;  tree_index++){

        // compute sensitivity
        params.delta_g = 3 * pow(params.l2_threshold, 2); // todo move out of loop
        params.delta_v = min((double) (params.l2_threshold / (1 + params.l2_lambda)),
                            2 * params.l2_threshold *
                            pow(1-params.learning_rate, tree_index));

        // init gradients for first tree of ensemble
        if(tree_index == 0){
            // do I really need to work on a copy?
            // don't think so; using an array to mark deleted rows
        }
        
        // compute number of training instances per tree
        if(params.balance_partition){  // TODO this part should be done outside loop
            // perfect split
            int remainder = train_set->length % params.nb_trees;
            if(remainder == 0)
                int number_of_rows = train_set->length / params.nb_trees;
            else {

            }
        } else {
            throw runtime_error("non-balanced resp. textbook formula partition not implemented yet");
        }


    }

    trees = {};
}