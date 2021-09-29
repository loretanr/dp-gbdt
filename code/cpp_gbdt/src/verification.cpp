#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>
#include "verification.h"
#include "parameters.h"
#include "data.h"
#include "gbdt/dp_ensemble.h"
#include "dataset_parser.h"
#include "spdlog/spdlog.h"

/* 
    Verification:
    runs the model on various (small to medium size) datasets for 
    easy verification of correctness. intermediate values are written to
    verification_logfile.
*/

extern size_t cv_fold_index;
std::ofstream verification_logfile;


int Verification::main(int argc, char *argv[])
{
    // Set up logging for debugging
    spdlog::set_level(spdlog::level::err);
    spdlog::set_pattern("[%H:%M:%S] [%^%5l%$] %v");

    // store datasets and their corresponding parameters here
    std::vector<DataSet *> datasets;
    std::vector<ModelParams> parameters;

    // --------------------------------------
    // select dataset(s) here.
    // you can either append some ModelParams to parameters here (and pass 
    // "false" to the parsing function), or let the get_xy function
    // do that (it'll create and append some default ones to the vector)
    ModelParams params = create_default_params();
    params.privacy_budget = 0.5;
    params.nb_trees = 5;
    params.gradient_filtering = false;
    params.balance_partition = true;
    params.leaf_clipping = true;
    params.use_dp = true;

    params.use_grid = false;
    params.grid_borders = std::make_tuple(0,1);
    params.grid_step_size = 0.001;
    params.scale_X = false;
    params.scale_X_percentile = 95;
    params.scale_X_privacy_budget = 0.4;

    // parameters.push_back(params);
    // datasets.push_back(Parser::get_abalone(parameters, 300, false)); // full abalone
    // parameters.push_back(params);
    // datasets.push_back(Parser::get_YearPredictionMSD(parameters, 150, false)); // small yearMSD
    parameters.push_back(params);
    datasets.push_back(Parser::get_adult(parameters, 320, false)); // small adult
    // parameters.push_back(params);
    // datasets.push_back(Parser::get_abalone(parameters, 4177, false)); // full abalone
    // --------------------------------------

    // use_dp (in combination with VERIFICATION_MODE, which disables dataset shuffling
    // in create_cross_val_inputs and rounding at certain places) turns off randomness completely
    // -> we get completely deterministic runs that are comparable to the python output.

    // do verification on all added datasets
    for(size_t i=0; i<datasets.size(); i++) {
        srand(0);
        DataSet *dataset = datasets[i];
        ModelParams &param = parameters[i];

        // Set up logging for verification
        verification_logfile.open(fmt::format("verification_logs/{}.cpp.log", dataset->name));
        std::cout << dataset->name << std::endl;

        if(param.use_grid and param.scale_X) {
            param.privacy_budget -= param.scale_X_privacy_budget;
            (*dataset).scale_X_columns(param);
        }

        // do cross validation, always 5 fold for now
        std::vector<TrainTestSplit *> cv_inputs = create_cross_validation_inputs(dataset, 5);
        delete dataset;
        cv_fold_index = 0;

        for (auto split : cv_inputs) {

            if(params.scale_y){
                split->train.scale_y(param, -1, 1);
            }

            // train the model
            DPEnsemble ensemble = DPEnsemble(&param);
            ensemble.train(&split->train);
            
            // predict with the test set
            std::vector<double> y_pred = ensemble.predict(split->test.X);

            if(params.scale_y){
                inverse_scale_y(param, split->train.scaler, y_pred);
            }
            
            // compute score
            double score = param.task->compute_score(split->test.y, y_pred);

            std::cout << std::setprecision(9) << score << " " << std::flush;
            cv_fold_index++;
            delete split;
        } std::cout << std::endl;
    
        verification_logfile.close();
    }
    return 0;
}
