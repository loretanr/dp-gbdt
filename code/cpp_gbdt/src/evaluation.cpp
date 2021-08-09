#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <algorithm>
#include "parameters.h"
#include "evaluation.h"
#include "utils.h"
#include "gbdt/dp_ensemble.h"
#include "dataset_parser.h"
#include "data.h"
#include "spdlog/spdlog.h"

typedef std::chrono::steady_clock::time_point Timer;

/* 
    Evaluation
    - also uses threads, so we should compile with "make fast"
    - runs your dataset for different pb's and writes output to results/xy.csv
*/

int Evaluation::main(int argc, char *argv[])
{
    // Set up logging for debugging
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%H:%M:%S] [%^%5l%$] %v");

    std::vector<ModelParams> parameters;

    // --------------------------------------
    // define ModelParams here
    ModelParams current_params;
    current_params.nb_trees = 50;
    current_params.gradient_filtering = true;
    current_params.balance_partition = true;
    current_params.leaf_clipping = true;
    current_params.scale_y = false;
    parameters.push_back(current_params);
    // --------------------------------------
    // select 1 dataset here
    DataSet dataset = Parser::get_abalone(parameters, 5000, false); // full abalone
    // DataSet dataset = Parser::get_YearPredictionMSD(parameters, 10000, false);
    // --------------------------------------
    // select privacy budgets
    // Note: pb=0 takes much much longer than dp-trees, because we're always using all samples
    std::vector<double> budgets = {0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2,2.5,3,4,5,6,7,8,9,10};
    // --------------------------------------

    // output file
    std::string time_string = get_time_string();
    std::string outfile_name = fmt::format("results/{}_{}.csv", dataset.name, time_string);
    std::ofstream output;
    output.open(outfile_name);
    std::cout << "evaluation, writing results to " << outfile_name << std::endl;
    output << "dataset,nb_samples,nb_trees,use_dp,privacy_budget,mean,std" << std::endl;

    // run the evaluations
    for(auto budget : budgets) {
        ModelParams param = parameters[0];
        param.privacy_budget = budget;
        std::cout << dataset.name << " pb=" << budget << std::endl;

        // toggle use_dp if budget is 0
        param.use_dp = budget != 0;

        /* cross validation */

        // split the data for each fold
        std::vector<TrainTestSplit> cv_inputs = create_cross_validation_inputs(dataset, 5);
        Timer time_begin = std::chrono::steady_clock::now();
        
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
            threads[thread_id] = std::thread(&DPEnsemble::train, &ensembles[thread_id],
                &(cv_inputs[thread_id].train));
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

            if(param.scale_y){
                inverse_scale(param, split->train.scaler, y_pred);
            }

            // compute score            
            double score = param.task->compute_score(split->test.y, y_pred);
            std::cout << std::setprecision(9) << score << " " << std::flush;
            scores.push_back(score);
        } 

        // print elapsed time
        Timer time_end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds> (time_end - time_begin).count();
        std::cout << "  (" << std::fixed << std::setprecision(1) << elapsed/1000 << "s)" << std::endl;

        // write mean score to file
        double mean = compute_mean(scores);
        double stdev = compute_stdev(scores, mean);
        output << fmt::format("{},{},{},{},{},{},{}", dataset.name, dataset.length, param.nb_trees, param.use_dp,
            param.privacy_budget, mean, stdev) << std::endl;
    }

    output.close();
    return 0;
}