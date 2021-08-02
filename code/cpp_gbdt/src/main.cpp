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
    LOG_INFO("hello MA start");

    // Define model parameters
    // only reason to use a vector is because parser expects it
    std::vector<ModelParams> params;
    ModelParams current_params = create_default_params();

    // change model params here if required:
    // e.g. current_params.privacy_budget = 42;
    current_params.privacy_budget = 0.1;
    current_params.use_dp = true;
    params.push_back(current_params);

    // Choose your dataset
    Parser parser = Parser();
    DataSet dataset = parser.get_abalone(params, 5000, false);

    std::cout << dataset.name << std::endl;

    // create cross validation inputs
    std::vector<TrainTestSplit> cv_inputs = create_cross_validation_inputs(dataset, 5);

    // do cross validation
    std::vector<double> rmses;
    for (auto split : cv_inputs) {

        // we should fit the Scaler only on the training set, according to
        // https://datascience.stackexchange.com/questions/38395/standardscaler-before-and-after-splitting-data
        split.train.scale(-1, 1);

        DPEnsemble ensemble = DPEnsemble(&params[0]);
        ensemble.train(&split.train);
        
        // predict with the test set
        std::vector<double> y_pred = ensemble.predict(split.test.X);

        // invert the feature scaling (if necessary)
        inverse_scale(split.train.scaler, y_pred);

        // compute score
        double score = params[0].task->compute_score(split.test.y, y_pred);

        std::cout << score << " " << std::flush;
    } std::cout << std::endl;

    LOG_INFO("hello MA end");
}