


#include "dp_tree.h"
#include "dp_ensemble.h"
#include "utils.h"
#include "dataset_parser.h"
#include "verification.h"


int main(int argc, char** argv)
{
    // parse flags
    for(int i = 1; i < argc; i++){
		if ( !strcmp(argv[i], "--verify") ){
            RANDOMIZATION = false;
			VERIFICATION_MODE = true;
            return Verification::main(argc, argv);
		} else {
            RANDOMIZATION = false;
            VERIFICATION_MODE = false;
        } 
        /* else if( !strcmp(argv[i], "--abalone") ){
			cout << "ABALONE FLAAG" << endl;
            abalone = true;
		} */
	}

    // Set up logging for debugging and validation
    spdlog::set_level(spdlog::level::err);
    spdlog::set_pattern("[%H:%M:%S] [%^%5l%$] %v");
    LOG_INFO("hello MA start");

    // Define model parameters
    ModelParams parammmms;
    parammmms.nb_trees = 50;
    parammmms.max_depth = 6;
    parammmms.gradient_filtering = true;
    parammmms.privacy_budget = 0.1;

    Parser parser = Parser();
    DataSet dataset = parser.get_abalone(parammmms, false);

    // dataset.X = {{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15}};  // TODO remove
    // dataset.y = {9001,9002,9003,9004,9005};
    vector<double> rmses;


    vector<TrainTestSplit> cv_inputs = create_cross_validation_inputs(dataset, 5, false);

    for (auto split : cv_inputs) {

        // we should fit the Scaler only on the training set, according to
        // https://datascience.stackexchange.com/questions/38395/standardscaler-before-and-after-splitting-data
        // However this probably hurts DP, because y_test is then not guaranteed between [-1,1]
        // but to keep data exactly the same as sklearn i'll only do on train for now.
        split.train.scale(-1, 1);
        // split.test.scale(-1, 1);


        DPEnsemble ensemble = DPEnsemble(&parammmms);
        ensemble.train(&split.train);
        
        // compute score
        vector<double> y_pred = ensemble.predict(split.test.X);

        // invert the feature scale
        inverse_scale(split.train.scaler, y_pred);


        std::transform(split.test.y.begin(), split.test.y.end(), 
                y_pred.begin(), y_pred.begin(), std::minus<double>());
        std::transform(y_pred.begin(), y_pred.end(),
                y_pred.begin(), [](double &c){return std::pow(c,2);});
        double average = std::accumulate(y_pred.begin(),y_pred.end(), 0.0) / y_pred.size();
        double rmse = std::sqrt(average);

        rmses.push_back(rmse);
        cout << rmse << " " << std::flush;
    }



    cout << endl << "RMSEs: " << setprecision(9);
    for(auto elem : rmses) {
        cout << elem << " ";
    } cout << endl;

    

    LOG_INFO("hello MA end");
}