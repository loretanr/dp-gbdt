#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "dp_tree.h"
#include "dp_ensemble.h"
#include "spdlog/spdlog.h"
#include <spdlog/sinks/stdout_sinks.h>
#include "utils.h"


DataSet get_abalone(ModelParams &params)
{
    ifstream infile("data/abalone.data");
    string line;
    VVF X; // TODO think about row/columnwise what makes sense
    vector<float> y;

    params.cat_idx = {0}; // first column is categorical
    params.num_idx = {1,2,3,4,5,6,7};

    while (getline(infile, line,'\n')) {
        stringstream ss(line);
        vector<string> strings = split_string(line, ',');
        vector<float> X_row;
        if(strings[0] == "M"){ // first column is gender (M/F/I)
            X_row.push_back(1.0f);
        } else if (strings[0] == "F") {
            X_row.push_back(2.0f);
        } else {
            X_row.push_back(3.0f);
        }
        for(size_t i=1;i<strings.size()-1; i++){
            X_row.push_back(stof(strings[i]));
        }
        y.push_back(stof(strings.back()));
        X.push_back(X_row);
    }
    return DataSet(X, y);
}

int main()
{
    spdlog::set_level(spdlog::level::debug); // Set global log level to debug
    spdlog::set_pattern("[%H:%M:%S] [%^%5l%$] %v");

    LOG_INFO("hello MA start");

    ModelParams parammmms;
    parammmms.nb_trees = 50;
    parammmms.max_depth = 6;
    parammmms.gradient_filtering = true;
    parammmms.privacy_budget = 0.1;

    DataSet dataset = get_abalone(parammmms);

    // dataset.X = {{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15}};  // TODO remove
    // dataset.y = {9001,9002,9003,9004,9005};


    DPEnsemble ensemble = DPEnsemble(&parammmms);
    TrainTestSplit split = train_test_split_random(dataset, 0.80f, false); // empty test for now
    // we should fit the Scaler only on the training set, according to
    // https://datascience.stackexchange.com/questions/38395/standardscaler-before-and-after-splitting-data
    // However this probably hurts DP, because y_test is then not guaranteed between [-1,1]
    // but to keep data exactly the same as sklearn i'll only do on train for now.
    
    split.train.scale(-1, 1);
    // split.test.scale(-1, 1);


    ensemble.train(&split.train);
    
    // compute score
    vector<float> y_pred = ensemble.predict(split.test.X);

    // invert the feature scale
    inverse_scale(split.train.scaler, y_pred);


    std::transform(split.test.y.begin(), split.test.y.end(), 
            y_pred.begin(), y_pred.begin(), std::minus<float>());
    std::transform(y_pred.begin(), y_pred.end(),
            y_pred.begin(), [](float &c){return std::pow(c,2);});
    float average = std::accumulate(y_pred.begin(),y_pred.end(), 0.0) / y_pred.size();
    float rmse = std::sqrt(average);

    cout << "RMSE: " << rmse << endl;

    

    LOG_INFO("hello MA end");
}