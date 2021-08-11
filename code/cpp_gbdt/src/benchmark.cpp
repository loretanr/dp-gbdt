#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include "parameters.h"
#include "benchmark.h"
#include "gbdt/dp_ensemble.h"
#include "dataset_parser.h"
#include "data.h"
#include "spdlog/spdlog.h"

/* 
    Benchmark:
    in order for this code to be fast, it has to be compiled with agressive 
    optimization flags (-> use "make clean; make fast")
    - vectorization: code is written such that the compiler can do it's thing 
        - should work well for intel skylake or later
    - threading: each cv-fold get his own thread
    - spicy speedup (>1200x measured compared to python)
*/

int Benchmark::main(int argc, char *argv[])
{
    // Set up logging for debugging
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%H:%M:%S] [%^%5l%$] %v");

    // store datasets and their corresponding parameters here
    std::vector<DataSet> datasets;
    std::vector<ModelParams> parameters;

    // --------------------------------------
    // select dataset(s) here.
    // you can either append some ModelParams to parameters here (and pass 
    // "false" to the parsing function), or let the get_xy function
    // do that (it'll create and append some default ones to the vector)
    ModelParams params;
    params.use_dp = true;
    params.privacy_budget = 6;
    params.nb_trees = 10;
    parameters.push_back(params);
    datasets.push_back(Parser::get_abalone(parameters, 5000, false)); // full abalone
    // datasets.push_back(Parser::get_YearPredictionMSD(parameters, 10000, true)); // medium yearMSD
    // datasets.push_back(Parser::get_adult(parameters, 4000, true)); // medium adult
    // --------------------------------------

    for(size_t i=0; i<datasets.size(); i++) {
        DataSet &dataset = datasets[i];
        ModelParams &param = parameters[i];
        std::cout << dataset.name << std::endl;

        /* threaded cross validation */

        // split the data for each fold
        std::vector<TrainTestSplit> cv_inputs = create_cross_validation_inputs(dataset, 5);
        std::chrono::steady_clock::time_point time_begin = std::chrono::steady_clock::now();
        
        // prepare the ressources for each thread
        std::vector<std::thread> threads(cv_inputs.size());
        std::vector<DPEnsemble> ensembles;
        for (auto &split : cv_inputs) {
            if(param.scale_y){
                split.train.scale(param, -1, 1);
            }
            ensembles.push_back(DPEnsemble(&param) );
        }

        // threads start training on ther respective folds
        for(size_t thread_id=0; thread_id<threads.size(); thread_id++){
            threads[thread_id] = std::thread(&DPEnsemble::train, &ensembles[thread_id], &(cv_inputs[thread_id].train));
        }
        for (auto &thread : threads) {
            thread.join(); // join once done
        }

        /* compute scores */

        for (size_t ensemble_id = 0; ensemble_id < ensembles.size(); ensemble_id++) {
            DPEnsemble *ensemble = &ensembles[ensemble_id];
            TrainTestSplit *split = &cv_inputs[ensemble_id];
            
            // predict with the test set
            std::vector<double> y_pred = ensemble->predict(split->test.X);

            if(param.scale_y){
                inverse_scale(param, split->train.scaler, y_pred);
            }

            // compute score            
            double score = param.task->compute_score(split->test.y, y_pred);
            std::cout << std::setprecision(9) << score << " " << std::flush;
        } 

        // print elapsed time (for 5 fold cv)
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds> (time_end - time_begin).count();
        std::cout << "  (" << std::fixed << std::setprecision(1) << elapsed/1000 << "s)" << std::endl;
    }
    return 0;
}
