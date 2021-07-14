#include "verification.h"

/* 
    runs model on various (smaller size) datasets for 
    easy verification of correctness
*/

std::ofstream verification_logfile;


int Verification::main(int argc, char *argv[])
{
    // Set up logging for debugging
    spdlog::set_level(spdlog::level::err);
    spdlog::set_pattern("[%H:%M:%S] [%^%5l%$] %v");

    // Define model parameters
    ModelParams parammmms;
    parammmms.nb_trees = 50;
    parammmms.max_depth = 6;
    parammmms.gradient_filtering = true;
    parammmms.leaf_clipping = true;
    parammmms.privacy_budget = 0.1;

    vector<DataSet> datasets;

    Parser parser = Parser();
    // datasets.push_back(parser.get_abalone(parammmms, true)); // small abalone (300)
    // datasets.push_back(parser.get_abalone(parammmms)); // full abalone (5000)
    datasets.push_back(parser.get_YearPredictionMSD(parammmms, true)); // small yearMSD (300)
    // datasets.push_back(parser.get_YearPredictionMSD(parammmms, false)); // medium yearMSD (800)

    for(auto &dataset : datasets) {

        // Set up logging for verification
        verification_logfile.open(fmt::format("verification_logs/{}.cpp.log", dataset.name));
        cout << dataset.name << endl;

        // do cross validation
        vector<double> rmses;
        vector<TrainTestSplit> cv_inputs = create_cross_validation_inputs(dataset, 5, false);
        cv_fold_index = 0;

        for (auto split : cv_inputs) {

            split.train.scale(-1, 1);
            DPEnsemble ensemble = DPEnsemble(&parammmms);
            ensemble.train(&split.train);
            
            // compute score
            vector<double> y_pred = ensemble.predict(split.test.X);

            // invert the feature scale
            inverse_scale(split.train.scaler, y_pred);

            // compute RMSE
            std::transform(split.test.y.begin(), split.test.y.end(), y_pred.begin(), y_pred.begin(), std::minus<double>());
            std::transform(y_pred.begin(), y_pred.end(), y_pred.begin(), [](double &c){return std::pow(c,2);});
            double average = std::accumulate(y_pred.begin(),y_pred.end(), 0.0) / y_pred.size();
            double rmse = std::sqrt(average);

            rmses.push_back(rmse);
            cout << setprecision(9) << rmse << " " << std::flush;
            cv_fold_index++;
        } cout << endl;
    
        verification_logfile.close();
    }

    return 0;
}
