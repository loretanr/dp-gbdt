#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "parameters.h"
#include "gbdt/dp_ensemble.h"
#include "data.h"

int main(int argc, char** argv)
{
    // seed randomness once and for all
    srand(time(NULL));

    // Define model parameters
    // reason to use a vector is because parser expects it
    std::vector<ModelParams> parameters;
    ModelParams current_params = create_default_params();

    // change model params here if required:
    current_params.privacy_budget = 10;
    current_params.nb_trees = 5;
    current_params.use_dp = true;
    current_params.gradient_filtering = true;
    current_params.balance_partition = true;
    current_params.leaf_clipping = false;
    current_params.scale_y = false;
    parameters.push_back(current_params);

    // Choose your dataset
    DataSet *dataset;

    std::cout << dataset->name << std::endl;

    // create cross validation inputs
    std::vector<TrainTestSplit *> cv_inputs = create_cross_validation_inputs(dataset, 5);
    delete dataset;

    // do cross validation
    std::vector<double> rmses;
    for (auto split : cv_inputs) {
        ModelParams params = parameters[0];

        if(params.scale_y){
            split->train.scale(params, -1, 1);
        }

        DPEnsemble ensemble = DPEnsemble(&params);
        ensemble.train(&split->train);
        
        // predict with the test set
        std::vector<double> y_pred = ensemble.predict_ensemble(split->test.X);

        if(params.scale_y) {
            inverse_scale(params, split->train.scaler, y_pred);
        }

        // compute score
        double score = params.task->compute_score(split->test.y, y_pred);

        std::cout << score << " " << std::flush;
        delete split;
    } std::cout << std::endl;
}