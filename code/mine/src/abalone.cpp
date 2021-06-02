#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "dp_tree.h"
#include "dp_ensemble.h"
#include "utils.h"

using namespace std;

DataSet get_abalone()
{
    ifstream infile("data/abalone.data");
    string line;
    float val;
    vector<vector<float>> X; // TODO think about row/columnwise what makes sense
    vector<float> y;

    int lineIdx = 0;
    while (getline(infile, line,'\n')) {
        stringstream ss(line);
        int colIdx = 0;
        vector<float> X_row;
        while(ss >> val) {
            if(colIdx < 8){
                X_row.push_back(val);
            } else {
                y.push_back(val);
            }
            colIdx++;
        }
        X.push_back(X_row);
        lineIdx++;
    }
    // dataset is just 2 pointers
    DataSet dataset = DataSet(X, y);
    return dataset;
}

int main()
{
    DataSet dataset = get_abalone();

    dataset.X = {{1,2,3},{4,5,6},{7,8,9}};  // TODO remove
    dataset.y = {11,12,13};

    ModelParams parammmms;
    parammmms.delta_g = 0.42;
    parammmms.use_bfs = true;
    parammmms.max_depth = 600000;

    DPEnsemble ensemble = DPEnsemble(&parammmms);
    TrainTestSplit split = train_test_split_random(dataset);
    //ensemble.train();

    cout << "hello MA world" << endl;
}