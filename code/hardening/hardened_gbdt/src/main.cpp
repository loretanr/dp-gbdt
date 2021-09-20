#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include "logging.h"
#include "utils.h"
#include "parameters.h"
#include "gbdt/dp_ensemble.h"
#include "dataset_parser.h"
#include "data.h"
#include "verification.h"
#include "spdlog/spdlog.h"

extern bool VERIFICATION_MODE;


int main(int argc, char** argv)
{
    // seed randomness once and for all
    srand(time(NULL));

    // parse flags, currently supporting "--verify"
    if(argc != 1){
        for(int i = 1; i < argc; i++){
            if ( ! std::strcmp(argv[i], "--verify") ){
                // go into verification mode
                VERIFICATION_MODE = true;
                return Verification::main(argc, argv);
            } else {
                throw std::runtime_error("unkown command line flag encountered");
            } 
        }
    } else { // no flags given, continue in this file
        VERIFICATION_MODE = false;
    }

    // Set up logging
    spdlog::set_level(spdlog::level::err);
    spdlog::set_pattern("[%H:%M:%S] [%^%5l%$] %v");

    // Define model parameters
    // reason to use a vector is because parser expects it
    std::vector<ModelParams> parameters;
    ModelParams current_params = create_default_params();

    // change model params here if required:
    current_params.privacy_budget = 10;
    current_params.nb_trees = 5;
    current_params.gradient_filtering = TRUE;
    current_params.balance_partition = TRUE;
    current_params.leaf_clipping = FALSE;
    current_params.scale_y = FALSE;
    parameters.push_back(current_params);

    // Choose your dataset
    DataSet *dataset = Parser::get_abalone(parameters, 300, false);

    std::cout << dataset->name << std::endl;
    std::chrono::steady_clock::time_point time_begin = std::chrono::steady_clock::now();

    // create cross validation inputs
    std::vector<TrainTestSplit *> cv_inputs = create_cross_validation_inputs(dataset, 5);
    delete dataset;

    // do cross validation
    std::vector<double> rmses;
    for (auto split : cv_inputs) {
        ModelParams params = parameters[0];

        if(is_true(params.scale_y)){
            split->train.scale(params, -1, 1);
        }

        DPEnsemble ensemble = DPEnsemble(&params);
        ensemble.train(&split->train);
        
        // predict with the test set
        std::vector<double> y_pred = ensemble.predict(split->test.X);

        if(is_true(params.scale_y)) {
            inverse_scale(params, split->train.scaler, y_pred);
        }

        // compute score
        double score = params.task->compute_score(split->test.y, y_pred);

        std::cout << score << " " << std::flush;
        delete split;
    } std::cout << std::endl;

    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds> (time_end - time_begin).count();
    std::cout << "(" << std::fixed << std::setprecision(1) << elapsed/1000 << "s)" << std::endl;
}