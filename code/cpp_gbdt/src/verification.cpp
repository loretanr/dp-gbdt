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
    std::vector<DataSet> datasets;
    std::vector<ModelParams> parameters;

    // --------------------------------------
    // select dataset(s) here.
    // you can either append some ModelParams to parameters here (and pass 
    // "false" to the parsing function), or let the get_xy function
    // do that (it'll create and append some default ones to the vector)
    Parser parser = Parser();
    datasets.push_back(parser.get_abalone(parameters, 320, true)); // small abalone
    datasets.push_back(parser.get_abalone(parameters, 4177, true)); // full abalone
    datasets.push_back(parser.get_YearPredictionMSD(parameters, 300, true)); // small yearMSD
    // datasets.push_back(parser.get_YearPredictionMSD(parameters, 1000, true)); // medium yearMSD
    datasets.push_back(parser.get_adult(parameters, 320, true)); // small adult
    // datasets.push_back(parser.get_adult(parameters, 1000, true)); // medium adult
    // --------------------------------------

    // use_dp (in combination with VERIFICATION_MODE, which disables dataset shuffling
    // in create_cross_val_inputs and rounding at certain places) turns off randomness completely
    // -> we get completely deterministic runs that are comparable to the python output.
    for(auto &elem : parameters){
        elem.use_dp = false;
    }

    // do verification on all added datasets
    for(size_t i=0; i<datasets.size(); i++) {
        DataSet &dataset = datasets[i];
        ModelParams &param = parameters[i];

        // Set up logging for verification
        verification_logfile.open(fmt::format("verification_logs/{}.cpp.log", dataset.name));
        std::cout << dataset.name << std::endl;

        // do cross validation, always 5 fold for now
        std::vector<TrainTestSplit> cv_inputs = create_cross_validation_inputs(dataset, 5);
        cv_fold_index = 0;

        for (auto split : cv_inputs) {

            // scale the features (y) to [-1,1] if necessary
            split.train.scale(-1, 1);

            // train the model
            DPEnsemble ensemble = DPEnsemble(&param);
            ensemble.train(&split.train);
            
            // predict with the test set
            std::vector<double> y_pred = ensemble.predict(split.test.X);

            // invert the feature scale (if necessary)
            inverse_scale(split.train.scaler, y_pred);
            
            // compute score
            double score = param.task->compute_score(split.test.y, y_pred);

            std::cout << std::setprecision(9) << score << " " << std::flush;
            cv_fold_index++;
        } std::cout << std::endl;
    
        verification_logfile.close();
    }
    return 0;
}
