#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "logging.h"
#include "parameters.h"
#include "gbdt/dp_ensemble.h"
#include "dataset_parser.h"
#include "data.h"
#include "verification.h"
#include "benchmark.h"
#include "evaluation.h"
#include "spdlog/spdlog.h"

extern bool VERIFICATION_MODE;


int main(int argc, char** argv)
{
    // seed randomness once and for all
    srand(time(NULL));

    // parse flags, currently supporting "--verify", "--bench", "--eval"
    if(argc != 1){
        for(int i = 1; i < argc; i++){
            if ( ! std::strcmp(argv[i], "--verify") ){
                // go into verification mode
                VERIFICATION_MODE = true;
                return Verification::main(argc, argv);
            } else if ( ! std::strcmp(argv[i], "--bench") ){
                // go into benchmark mode
                VERIFICATION_MODE = false;
                return Benchmark::main(argc, argv);
            } else if ( ! std::strcmp(argv[i], "--eval") ){
                // go into evaluation mode
                VERIFICATION_MODE = false;
                return Evaluation::main(argc, argv); 
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
    current_params.nb_trees = 30;
    current_params.use_dp = true;
    current_params.learning_rate = 0.1;
    current_params.gradient_filtering = false;
    current_params.balance_partition = true;
    current_params.leaf_clipping = true;
    current_params.scale_y = false;

    current_params.use_grid = false;
    current_params.grid_borders = std::make_tuple(0,1);
    current_params.grid_step_size = 0.001;
    current_params.scale_X = false;
    current_params.scale_X_percentile = 95;
    current_params.scale_X_privacy_budget = 0.4;

    parameters.push_back(current_params);

    // Choose your dataset
    // DataSet *dataset = Parser::get_abalone(parameters, 5000, false);
    // DataSet *dataset = Parser::get_bcw(parameters, 700, false);
    DataSet *dataset = Parser::get_YearPredictionMSD(parameters, 17000, false);

    std::cout << dataset->name << std::endl;
    // int count1 = std::count_if(dataset->y.begin(), dataset->y.end(),
    //     [](double c){ return c == 0.0; });
    // int count2 = std::count_if(dataset->y.begin(), dataset->y.end(),
    //     [](double c){ return c == 1.0; });
    // std::cout << "y 0's (" << count1 << ") y 1's (" << count2 << ")" << std::endl;
    // std::cout << "-> zeroR = " << std::setprecision(4) << (double) std::max(count1, count2) / (double) dataset->length << std::endl;

    ModelParams params = parameters[0];

    if(params.use_grid and params.scale_X) {
        params.privacy_budget -= params.scale_X_privacy_budget;
        (*dataset).scale_X_columns(params);
    }

    // if 5-fold cv itself is not consistent enough, can run it multiple times
    int NUM_REPS = 1;

    std::vector<double> scores;
    for(int i=0; i<NUM_REPS; i++){

        // create cross validation inputs
        std::vector<TrainTestSplit *> cv_inputs = create_cross_validation_inputs(dataset, 5);

        // do cross validation
        for (auto split : cv_inputs) {
            
            if(params.scale_y){
                split->train.scale_y(params, -1, 1);
            }

            DPEnsemble ensemble = DPEnsemble(&params);
            ensemble.train(&split->train);
            
            // predict with the test set
            std::vector<double> y_pred = ensemble.predict(split->test.X);

            if(params.scale_y) {
                inverse_scale_y(params, split->train.scaler, y_pred);
            }

            // compute score
            double score = params.task->compute_rmse(split->test.y, y_pred);

            std::cout << score << " " << std::flush;
            scores.push_back(score);
            delete split;
        } std::cout << std::endl;
    }
    // average score
    std::cout << "total avg score (" << scores.size() << " ensembles): " << std::fixed << std::setprecision(4) <<
        std::accumulate(scores.begin(), scores.end(), 0.0) / (double) scores.size() << std::endl;
}