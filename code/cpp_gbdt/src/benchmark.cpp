#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include "benchmark.h"
#include "dp_ensemble.h"
#include "dataset_parser.h"
#include "spdlog/spdlog.h"

/* 
    Benchmark:
    Compare python and cpp runtimes
*/

int Benchmark::main(int argc, char *argv[])
{
    // Set up logging for debugging
    spdlog::set_level(spdlog::level::err);
    spdlog::set_pattern("[%H:%M:%S] [%^%5l%$] %v");

    // store datasets and their corresponding parameters here
    std::vector<DataSet> datasets;
    std::vector<ModelParams> parameters;

    // select dataset(s) here. So far only abalone and year work correctly
    // you can either append some ModelParams to parameters here, or let 
    // the get_xy function do that (it'll create and append some default ones)

    Parser parser = Parser();
    datasets.push_back(parser.get_abalone(parameters, 4177, true)); // full abalone
    datasets.push_back(parser.get_adult(parameters, 4000, true)); // medium adult

    for(size_t i=0; i<datasets.size(); i++) {
        DataSet &dataset = datasets[i];
        ModelParams &param = parameters[i];

        // Set up logging for verification
        std::cout << dataset.name << std::endl;

        // do cross validation
        std::vector<double> scores;
        std::vector<TrainTestSplit> cv_inputs = create_cross_validation_inputs(dataset, 5, false);
        cv_fold_index = 0;
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();


        for (auto split : cv_inputs) {

            // scale the features (y) to [-1,1] if necessary
            split.train.scale(-1, 1);

            // train the ensemble
            DPEnsemble ensemble = DPEnsemble(&param);
            ensemble.train(&split.train);
            
            // compute score
            std::vector<double> y_pred = ensemble.predict(split.test.X);

            // invert the feature scale (if necessary)
            inverse_scale(split.train.scaler, y_pred);

            // compute score            
            double score = param.lossfunction->compute_score(split.test.y, y_pred);
            scores.push_back(score);
            std::cout << std::setprecision(9) << score << " " << std::flush;
            cv_fold_index++;
        } 
        // stop time (for 5 fold cv)
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count();
        std::cout << "  (" << std::fixed << std::setprecision(1) << elapsed/1000 << "s)" << std::endl;
    }
    return 0;
}
