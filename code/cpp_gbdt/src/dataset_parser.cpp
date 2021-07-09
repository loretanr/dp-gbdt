#include "dataset_parser.h"

DataSet Parser::get_abalone(ModelParams &params)
{
    ifstream infile("datasets/real/abalone.data");
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