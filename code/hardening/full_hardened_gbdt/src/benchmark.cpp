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
#include "benchmark.h"
#include "spdlog/spdlog.h"


int Benchmark::main(int argc, char** argv)
{
    // Set up logging
    spdlog::set_level(spdlog::level::err);
    spdlog::set_pattern("[%H:%M:%S] [%^%5l%$] %v");

    // Define model parameters
    // reason to use a vector is because parser expects it
    std::vector<DataSet *> datasets;
    std::vector<ModelParams> parameters;

    // change model params here if required:
    ModelParams params;
    params.use_grid = FALSE;
    params.privacy_budget = 0.5;
    params.nb_trees = 10;
    params.leaf_clipping = TRUE;
    params.balance_partition = TRUE;
    params.gradient_filtering = FALSE;
    params.min_samples_split = 2;
    params.learning_rate = 0.1;
    params.max_depth = 6;
    params.scale_X = FALSE;

    parameters.push_back(params);
    datasets.push_back(Parser::get_abalone(parameters, 300, false));
    parameters.push_back(params);
    datasets.push_back(Parser::get_abalone(parameters, 1000, false));
    parameters.push_back(params);
    datasets.push_back(Parser::get_abalone(parameters, 4177, false));
    parameters.push_back(params);
    datasets.push_back(Parser::get_adult(parameters, 300, false));
    parameters.push_back(params);
    datasets.push_back(Parser::get_adult(parameters, 1000, false));
    parameters.push_back(params);
    datasets.push_back(Parser::get_adult(parameters, 5000, false));
    parameters.push_back(params);
    datasets.push_back(Parser::get_YearPredictionMSD(parameters, 300, false));
    parameters.push_back(params);
    datasets.push_back(Parser::get_YearPredictionMSD(parameters, 1000, false));

    for(size_t i=0; i<datasets.size(); i++) {

        DataSet *dataset = datasets[i];
        ModelParams &param = parameters[i];
        std::cout << dataset->name << std::endl;

        if(is_true(param.use_grid) and is_true(param.scale_X)) {
            param.privacy_budget -= param.scale_X_privacy_budget;
            (*dataset).scale_X_columns(param);
        }

        /* threaded cross validation */

        std::vector<TrainTestSplit *> cv_inputs = create_cross_validation_inputs(dataset, 5);
        delete dataset;

        std::chrono::steady_clock::time_point time_begin = std::chrono::steady_clock::now();
        
        for (auto split : cv_inputs) {

            if(is_true(param.scale_y)){
                split->train.scale_y(param, -1, 1);
            }

            DPEnsemble ensemble = DPEnsemble(&param);
            ensemble.train(&split->train);
            
            // predict with the test set
            std::vector<double> y_pred = ensemble.predict(split->test.X);

            if(is_true(param.scale_y)) {
                inverse_scale_y(param, split->train.scaler, y_pred);
            }

            // compute score
            double score = param.task->compute_score(split->test.y, y_pred);

            std::cout << score << " " << std::flush;
            delete split;
        }
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds> (time_end - time_begin).count();
        std::cout << "(" << std::fixed << std::setprecision(1) << elapsed/1000 << "s)" << std::endl;
    }
    return 0;
}