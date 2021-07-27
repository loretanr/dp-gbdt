#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "dp_ensemble.h"
#include "dataset_parser.h"
#include "verification.h"
#include "benchmark.h"
#include "spdlog/spdlog.h"



int main(int argc, char** argv)
{
    // parse flags
    for(int i = 1; i < argc; i++){
		if ( ! std::strcmp(argv[i], "--verify") ){
            // go into verification mode -> run model on small datasets
            RANDOMIZATION = false;
			VERIFICATION_MODE = true;
            return Verification::main(argc, argv);
		} else if ( ! std::strcmp(argv[i], "--bench") ){
            // go into benchmark mode
            RANDOMIZATION = true;
			VERIFICATION_MODE = false;
            return Benchmark::main(argc, argv);
		} else {
            // Keep RAMDOMIZATION off for now
            // otherwise impossible to verify algorithm while writing code
            RANDOMIZATION = false;      
            VERIFICATION_MODE = false;
        } 
	}

    // Set up logging
    spdlog::set_level(spdlog::level::err);
    spdlog::set_pattern("[%H:%M:%S] [%^%5l%$] %v");
    LOG_INFO("hello MA start");

    // Define model parameters
    std::vector<ModelParams> params;
    ModelParams current_params = create_default_params();

    // change current params here if required:
    // current_params.privacy_budget = 42;
    params.push_back(current_params);

    Parser parser = Parser();
    DataSet dataset = parser.get_abalone(params, 300, false);

    // create cross validation inputs
    std::vector<TrainTestSplit> cv_inputs = create_cross_validation_inputs(dataset, 5, false);

    // do cross validation
    std::vector<double> rmses;
    for (auto split : cv_inputs) {

        // we should fit the Scaler only on the training set, according to
        // https://datascience.stackexchange.com/questions/38395/standardscaler-before-and-after-splitting-data
        split.train.scale(-1, 1);

        DPEnsemble ensemble = DPEnsemble(&params[0]);
        ensemble.train(&split.train);
        
        // compute score
        std::vector<double> y_pred = ensemble.predict(split.test.X);

        // invert the feature scaling
        inverse_scale(split.train.scaler, y_pred);

        // compute RMSE
        std::transform(split.test.y.begin(), split.test.y.end(), 
                y_pred.begin(), y_pred.begin(), std::minus<double>());
        std::transform(y_pred.begin(), y_pred.end(),
                y_pred.begin(), [](double &c){return std::pow(c,2);});
        double average = std::accumulate(y_pred.begin(),y_pred.end(), 0.0) / y_pred.size();
        double rmse = std::sqrt(average);

        rmses.push_back(rmse);
        std::cout << rmse << " " << std::flush;
    }

    std::cout << std::endl << "RMSEs: " << std::setprecision(9);
    for(auto elem : rmses) {
        std::cout << elem << " ";
    } std::cout << std::endl;

    LOG_INFO("hello MA end");
}