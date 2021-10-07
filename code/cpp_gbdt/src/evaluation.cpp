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
    spdlog::set_level(spdlog::level::err);
    spdlog::set_pattern("[%H:%M:%S] [%^%5l%$] %v");

    std::vector<ModelParams> parameters;

    // --------------------------------------
    // define ModelParams here
    ModelParams current_params;
    current_params.nb_trees = 20;
    current_params.leaf_clipping = false;
    current_params.balance_partition = true;
    current_params.gradient_filtering = true;
    current_params.min_samples_split = 2;
    current_params.learning_rate = 0.1;
    current_params.max_depth = 6;
    current_params.use_grid = false;
    current_params.scale_X = false;

    parameters.push_back(current_params);
    // --------------------------------------
    // select 1 dataset here
    // DataSet *dataset = Parser::get_abalone(parameters, 5000, false); // full abalone
    // DataSet *dataset = Parser::get_adult(parameters, 5000, false);
    DataSet *dataset = Parser::get_YearPredictionMSD(parameters, 10000, false);
    // --------------------------------------
    // select privacy budgets
    // Note: pb=0 takes much much longer than dp-trees, because we're always using all samples
    std::vector<double> budgets = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2,2.5,3,4,5,6,7,8,9,10};
    // --------------------------------------

    // output file
    std::string time_string = get_time_string();
    std::string dataset_name = dataset->name;
    int dataset_length = dataset->length;
    std::string outfile_name = fmt::format("results/year/{}_{}.csv", dataset_name, time_string);
    std::ofstream output;
    output.open(outfile_name);
    std::cout << "evaluation, writing results to " << outfile_name << std::endl;
    output << "dataset,nb_samples,nb_trees,use_dp,privacy_budget,mean,std,glc,gdf,mean_mape,std_mape" << std::endl;


    double summm = std::accumulate(dataset->y.begin(), dataset->y.end(), 0.0);
    std::cout << "average y of training set: " << summm / dataset->y.size() << std::endl;


    // currently we use the same folds for all budgets. Not sure whether that's good or bad.

    ModelParams param = parameters[0];

    if(param.use_grid and param.scale_X) {
            param.privacy_budget -= param.scale_X_privacy_budget;
            (*dataset).scale_X_columns(param);
        }

    // run the evaluations
    for(auto budget : budgets) {
        param.privacy_budget = budget;
        param.use_dp = budget != 0.;
        std::cout << dataset_name << " pb=" << budget << std::endl;

        std::vector<TrainTestSplit *> cv_inputs = create_cross_validation_inputs(dataset, 5);

        // YEAR MSD SHIT
        for(auto split : cv_inputs){
            split->test = *(Parser::get_YearPredictionMSD_test(parameters, split->test.length, false));
        }

        Timer time_begin = std::chrono::steady_clock::now();
        
        // prepare the ressources for each thread
        std::vector<std::thread> threads(cv_inputs.size());
        std::vector<DPEnsemble> ensembles;
        for (auto split : cv_inputs) {
            if(param.scale_y){
                split->train.scale_y(param, -1, 1);
            }
            ensembles.push_back(DPEnsemble(&param) );
        }

        // threads start training on ther respective folds
        for(size_t thread_id=0; thread_id<threads.size(); thread_id++){
            threads[thread_id] = std::thread(&DPEnsemble::train, &ensembles[thread_id],
                &(cv_inputs[thread_id]->train));
        }
        for (auto &thread : threads) {
            thread.join(); // join once done
        }

        /* compute scores */

        std::vector<double> scores_rmse;
        std::vector<double> scores_mape;
        for (size_t ensemble_id = 0; ensemble_id < ensembles.size(); ensemble_id++) {
            DPEnsemble *ensemble = &ensembles[ensemble_id];
            TrainTestSplit *split = cv_inputs[ensemble_id];
            
            // predict with the test set
            std::vector<double> y_pred = ensemble->predict(split->test.X);

            if(param.scale_y){
                inverse_scale_y(param, split->train.scaler, y_pred);
            }

            // THIS WILL GET YOU THE BASELINE ("PREDICT WITH MEAN")
            // for(auto &elem : y_pred) {
            //     elem = summm / dataset->y.size();
            // }

            // compute score
            double score_rmse = param.task->compute_rmse(split->test.y, y_pred);
            double score_mape = param.task->compute_mape(split->test.y, y_pred);

            std::cout << std::fixed << std::setprecision(3) << "(" << score_rmse << "," << score_mape << ") " << std::flush;
            scores_rmse.push_back(score_rmse);
            scores_mape.push_back(score_mape);
            delete split;
        } 

        // print elapsed time
        Timer time_end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds> (time_end - time_begin).count();
        std::cout << "  (" << std::fixed << std::setprecision(1) << elapsed/1000 << "s)" << std::endl;

        // write mean score to file
        double mean_rmse = compute_mean(scores_rmse);
        double mean_mape = compute_mean(scores_mape);
        double stdev_rmse = compute_stdev(scores_rmse, mean_rmse);
        double stdev_mape = compute_stdev(scores_mape, mean_mape);
        output << fmt::format("{},{},{},{},{},{},{},{},{},{},{}", dataset_name, dataset_length, param.nb_trees, param.use_dp,
            param.privacy_budget, mean_rmse, stdev_rmse, param.leaf_clipping, param.gradient_filtering, mean_mape, stdev_mape) << std::endl;
    }

    delete dataset;

    output.close();
    return 0;
}
