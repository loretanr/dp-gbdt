#include "verification.h"

/* 
    runs model on various (smaller size) datasets for 
    easy verification of correctness
*/

std::ofstream verification_logfile;


int Verification::main(int argc, char *argv[])
{
    // Set up logging for debugging
    spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("[%H:%M:%S] [%^%5l%$] %v");

    vector<DataSet> datasets;
    vector<ModelParams> parameters;

    Parser parser = Parser();
    // datasets.push_back(parser.get_abalone(parameters, 300, true)); // small abalone
    // datasets.push_back(parser.get_abalone(parameters, 4177, true)); // full abalone
    // datasets.push_back(parser.get_YearPredictionMSD(parameters, 300, true)); // small yearMSD
    // datasets.push_back(parser.get_YearPredictionMSD(parameters, 1000, true)); // medium yearMSD
    datasets.push_back(parser.get_adult(parameters, 300, true)); // small adult

    for(size_t i=0; i<datasets.size(); i++) {
        DataSet &dataset = datasets[i];
        ModelParams &param = parameters[i];

        // Set up logging for verification
        verification_logfile.open(fmt::format("verification_logs/{}.cpp.log", dataset.name));
        cout << dataset.name << endl;

        // do cross validation
        vector<double> rmses;
        vector<TrainTestSplit> cv_inputs = create_cross_validation_inputs(dataset, 5, false);
        cv_fold_index = 0;

        for (auto split : cv_inputs) {

            split.train.scale(-1, 1);
            DPEnsemble ensemble = DPEnsemble(&param);
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
