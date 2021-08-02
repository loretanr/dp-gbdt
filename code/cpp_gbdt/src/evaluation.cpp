#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <algorithm>
#include "parameters.h"
#include "evaluation.h"
#include "gbdt/dp_ensemble.h"
#include "dataset_parser.h"
#include "data.h"
#include "spdlog/spdlog.h"

/* 
    Evaluation
        TODO
*/

int Evaluation::main(int argc, char *argv[])
{
    std::cout << "evaluation, writing results to xyz TODO" << std::endl;
    // Set up logging for debugging
    spdlog::set_level(spdlog::level::err);
    spdlog::set_pattern("[%H:%M:%S] [%^%5l%$] %v");

    std::vector<ModelParams> parameters;
    Parser parser = Parser();

    // --------------------------------------
    // define ModelParams here
    ModelParams params = create_default_params();
    parameters.push_back(params);
    // --------------------------------------
    // select 1 dataset here
    DataSet dataset = parser.get_abalone(parameters, 4177, false); // full abalone
    // --------------------------------------
    // select privacy budgets
    std::vector<double> budgets = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 2.5, 3, 4};
    // --------------------------------------

    // output file
    time_t t = time(0);   // get time now
    struct tm *now = localtime(&t);
    char buffer [80];
    strftime(buffer,80,"%m.%d_%H:%M",now);
    std::ofstream output;
    output.open(fmt::format("results/{}_{}", dataset.name, buffer));
    output << "dataset,nb_samples,privacy_budget,mean" << std::endl;

    for(auto budget : budgets) {
        ModelParams param = parameters[0];
        param.privacy_budget = budget;
        std::cout << dataset.name << " pb=" << budget << std::endl;

        /* cross validation */

        // split the data for each fold
        std::vector<TrainTestSplit> cv_inputs = create_cross_validation_inputs(dataset, 5);
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
        for(size_t thread_id=0; thread_id<threads.size(); thread_id++){
            threads[thread_id] = std::thread(&DPEnsemble::train, &ensembles[thread_id], &(cv_inputs[thread_id].train));
        }
        for (auto &thread : threads) {
            thread.join(); // join once done
        }

        /* compute scores */

        std::vector<double> scores;
        for (size_t ensemble_id = 0; ensemble_id < ensembles.size(); ensemble_id++) {
            DPEnsemble *ensemble = &ensembles[ensemble_id];
            TrainTestSplit *split = &cv_inputs[ensemble_id];
            
            // predict with the test set
            std::vector<double> y_pred = ensemble->predict(split->test.X);

            // invert the feature scale (if necessary)
            inverse_scale(split->train.scaler, y_pred);

            // compute score            
            double score = param.task->compute_score(split->test.y, y_pred);
            std::cout << std::setprecision(9) << score << " " << std::flush;
            scores.push_back(score);
        } 

        // print elapsed time (for 5 fold cv)
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds> (time_end - time_begin).count();
        std::cout << "  (" << std::fixed << std::setprecision(1) << elapsed/1000 << "s)" << std::endl;

        // write mean score to file
        double mean = std::accumulate(scores.begin(), scores.end(), 0.0) / 5;
        output << fmt::format("{},{},{},{}", dataset.name, dataset.length,
            param.privacy_budget, mean) << std::endl;
    }
    output.close();
    return 0;
}
