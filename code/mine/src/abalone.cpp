#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "dp_tree.h"
#include "dp_ensemble.h"
#include "utils.h"

using namespace std;

DataSet get_abalone(ModelParams &params)
{
    ifstream infile("data/abalone.data");
    string line;
    vector<vector<float>> X; // TODO think about row/columnwise what makes sense
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
    ModelParams parammmms;
    parammmms.nb_trees = 50;
    parammmms.delta_g = 0.42;
    parammmms.max_depth = 600000;

    DataSet dataset = get_abalone(parammmms);
    dataset.scale(parammmms, -1, 1);

    // dataset.X = {{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15}};  // TODO remove
    // dataset.y = {9001,9002,9003,9004,9005};


    DPEnsemble ensemble = DPEnsemble(&parammmms);
    TrainTestSplit split = train_test_split_random(dataset, 1); // empty test for now
    
    ensemble.train(&split.train);

    cout << "hello MA world" << endl;
}