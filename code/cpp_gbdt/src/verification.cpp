#include "verification.h"

/* 
    runs model on different (smaller size) datasets for 
    easy verificaation of correctness
*/
int Verification::main(int argc, char *argv[])
{
    cout << "cpp verification start" << endl;

    // Set up logging for debugging and validation
    spdlog::set_level(spdlog::level::err);
    spdlog::set_pattern("[%H:%M:%S] [%^%5l%$] %v");
    verification_logfile.open("validation_logs/cpp.log");

    // Define model parameters
    ModelParams parammmms;
    parammmms.nb_trees = 50;
    parammmms.max_depth = 6;
    parammmms.gradient_filtering = true;
    parammmms.privacy_budget = 0.1;

    Parser parser = Parser();
    DataSet dataset = parser.get_abalone(parammmms);

    vector<double> rmses;

    vector<TrainTestSplit> cv_inputs = create_cross_validation_inputs(dataset, 5, false);

    for (auto split : cv_inputs) {

        split.train.scale(-1, 1);

        DPEnsemble ensemble = DPEnsemble(&parammmms);
        ensemble.train(&split.train);
        
        // compute score
        vector<double> y_pred = ensemble.predict(split.test.X);

        // invert the feature scale
        inverse_scale(split.train.scaler, y_pred);


        // compute RMSE
        std::transform(split.test.y.begin(), split.test.y.end(), 
                y_pred.begin(), y_pred.begin(), std::minus<double>());
        std::transform(y_pred.begin(), y_pred.end(),
                y_pred.begin(), [](double &c){return std::pow(c,2);});
        double average = std::accumulate(y_pred.begin(),y_pred.end(), 0.0) / y_pred.size();
        double rmse = std::sqrt(average);

        rmses.push_back(rmse);
        cout << "CV fold x rmse: " << rmse << endl;
    }

    cout << "RMSEs: " << setprecision(9);
    for(auto elem : rmses) {
        cout << elem << " ";
    } cout << endl;
 
    verification_logfile.close();
}
