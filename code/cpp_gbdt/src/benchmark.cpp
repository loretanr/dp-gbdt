#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include "benchmark.h"
#include "dp_ensemble.h"
#include "dataset_parser.h"
#include "spdlog/spdlog.h"

#include <thread>

/* 
    Benchmark:
    - to be compiled with agressive optimization flags (use "make fast")
    - threading: each cv-fold get his own thread
    - spicy speedup (40-70x compared to python)
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
    datasets.push_back(parser.get_YearPredictionMSD(parameters, 4000, true)); // medium yearMSD
    datasets.push_back(parser.get_adult(parameters, 4000, true)); // medium adult

    for(size_t i=0; i<datasets.size(); i++) {
        DataSet &dataset = datasets[i];
        ModelParams &param = parameters[i];

        std::cout << dataset.name << std::endl;

        /* perform cross validation */

        // split the data for each fold
        std::vector<TrainTestSplit> cv_inputs = create_cross_validation_inputs(dataset, 5, false);
        std::chrono::steady_clock::time_point time_begin = std::chrono::steady_clock::now();
        
        // prepare the ressources for each thread
        std::vector<std::thread> threads(cv_inputs.size());
        std::vector<DPEnsemble> ensembles;
        for (auto &split : cv_inputs) {
            // scale the features (y) to [-1,1] if necessary
            split.train.scale(-1, 1);
            ensembles.push_back(DPEnsemble(&param) );
        }

        // threads start training on ther respective folds
        for(int thread_id=0; thread_id<threads.size(); thread_id++){
            threads[thread_id] = std::thread(&DPEnsemble::train, &ensembles[thread_id], &(cv_inputs[thread_id].train));
        }
        for (auto &thread : threads) {
            thread.join(); // join once done
        }

        /* compute scores */

        for (int ensemble_id = 0; ensemble_id < ensembles.size(); ensemble_id++) {

            DPEnsemble *ensemble = &ensembles[ensemble_id];
            TrainTestSplit *split = &cv_inputs[ensemble_id];
            
            std::vector<double> y_pred = ensemble->predict(split->test.X);

            // invert the feature scale (if necessary)
            inverse_scale(split->train.scaler, y_pred);

            // compute score            
            double score = param.lossfunction->compute_score(split->test.y, y_pred);
            std::cout << std::setprecision(9) << score << " " << std::flush;
        } 

        // print elapsed time (for 5 fold cv)
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds> (time_end - time_begin).count();
        std::cout << "  (" << std::fixed << std::setprecision(1) << elapsed/1000 << "s)" << std::endl;
    }
    return 0;
}
