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
    vector<double> y;

    params.cat_idx = {0}; // first column is categorical
    params.num_idx = {1,2,3,4,5,6,7};

    while (getline(infile, line,'\n')) {
        stringstream ss(line);
        vector<string> strings = split_string(line, ',');
        vector<double> X_row;
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


vector<TrainTestSplit> create_cross_validation_inputs(DataSet &dataset, int folds, bool shuffle)
{
    if(shuffle) {
        srand(time(0));
        random_shuffle(dataset.X.begin(), dataset.X.end());
        random_shuffle(dataset.y.begin(), dataset.y.end());
    }

    int fold_size = dataset.length / folds;
    vector<int> fold_sizes(folds, fold_size);
    int remainder = dataset.length % folds;
    int index = 0;
    while (remainder != 0) {
        fold_sizes[index++]++;
        remainder--;
    }
    // each entry marks the first element of a fold (to be used as test set at some point)
    deque<int> indices(folds);
    std::partial_sum(fold_sizes.begin(), fold_sizes.end(), indices.begin());
    indices.push_front(0); 
    indices.pop_back();

    vector<TrainTestSplit> splits;

    for(int i=0; i<folds; i++) {
        VVF X_copy = dataset.X;
        vector<double> y_copy = dataset.y;

        VVF::iterator x_iterator = X_copy.begin() + indices[i];
        vector<double>::iterator y_iterator = y_copy.begin() + indices[i];

        VVF x_test(x_iterator, x_iterator + fold_sizes[i]);
        vector<double> y_test(y_iterator, y_iterator + fold_sizes[i]);

        X_copy.erase(x_iterator, x_iterator + fold_sizes[i]);
        y_copy.erase(y_iterator, y_iterator + fold_sizes[i]);

        VVF x_train(X_copy.begin(), X_copy.end());
        vector<double> y_train(y_copy.begin(), y_copy.end());

        DataSet train(x_train,y_train);
        DataSet test(x_test, y_test);

        splits.push_back(TrainTestSplit(train,test));
    }
    return splits;
}


int main()
{
    // DISTRIBUTION test
    // double scale = 4;
    // vector<double> samples;
    // srand(time(NULL));
    // Laplace lap(scale, rand());
    // for(int i=0; i<10000; i++){
    //     samples.push_back(lap.return_a_random_variable(scale));
    // }
    // ofstream myfile;
    // myfile.open ("laplace.txt");
    // for(auto sample : samples) {
    //     myfile << sample << " ";
    // }
    // myfile.close();
    // exit(0);
    // =======================================================


    spdlog::set_level(spdlog::level::info); // Set global log level to debug
    
    spdlog::set_pattern("[%H:%M:%S] [%^%5l%$] %v");

    LOG_INFO("hello MA start");

    // int NUM_REPEATS = 5;

    ModelParams parammmms;
    parammmms.nb_trees = 50;
    parammmms.max_depth = 6;
    parammmms.gradient_filtering = true;
    parammmms.privacy_budget = 0.1;

    DataSet dataset = get_abalone(parammmms);

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
        cout << "CV fold x rmse: " << rmse << endl;
    }



    cout << "RMSEs: " << setprecision(9);
    for(auto elem : rmses) {
        cout << elem << " ";
    } cout << endl;

    

    LOG_INFO("hello MA end");
}