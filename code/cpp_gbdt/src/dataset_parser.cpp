#include "dataset_parser.h"

DataSet Parser::get_abalone(vector<ModelParams> &parameters, size_t num_samples, bool default_params)
{
    ifstream infile("datasets/real/abalone.data");
    string line;
    VVD X;
    vector<double> y;

    if (default_params) {
        ModelParams params = create_default_params();
        params.cat_idx = {0}; // first column is categorical
        params.num_idx = {1,2,3,4,5,6,7};
        parameters.push_back(params);
    } else {
        parameters.back().num_idx = {1,2,3,4,5,6,7};
        parameters.back().cat_idx = {0};
    }

    size_t current_index = 0;

    while (getline(infile, line,'\n') && current_index < num_samples) {
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
        current_index++;
    }
    
    DataSet dataset = DataSet(X,y);
    dataset.name = num_samples == 300 ? "abalone_small" : "abalone_full";
    return dataset;
}


DataSet Parser::get_YearPredictionMSD(vector<ModelParams> &parameters, size_t num_samples, bool default_params)
{
    ifstream infile("datasets/real/YearPredictionMSD.txt");
    string line;
    VVD X;
    vector<double> y;

    std::vector<int> v(90); // vector with 90 ints
    std::iota(std::begin(v), std::end(v), 0); // fill with numbers 0..89
    if (default_params){
        ModelParams params = create_default_params();
        params.cat_idx = {};
        params.num_idx = v;
        parameters.push_back(params);
    } else {
        parameters.back().cat_idx = {};
        parameters.back().num_idx = v;
    }

    size_t current_index = 0;

    while (getline(infile, line,'\n') && current_index < num_samples) {
        stringstream ss(line);
        vector<string> strings = split_string(line, ',');
        vector<double> X_row;
        for(size_t i=1;i<strings.size(); i++){
            X_row.push_back(stof(strings[i]));
        }
        y.push_back(stof(strings[0]));
        X.push_back(X_row);
        current_index++;
    }
    DataSet dataset = DataSet(X,y);
    dataset.name = num_samples == 300 ? "yearMSD_small" : "yearMSD_medium";
    return dataset;
}

DataSet Parser::get_adult(vector<ModelParams> &parameters, size_t num_samples = false, bool default_params = false)
{
    ifstream train_infile("datasets/real/adult.data");
    ifstream test_infile("datasets/real/adult.test");
    VVD X;
    vector<double> y;

    vector<int> numerical = {0,4,10,11,12};
    vector<int> categorical = {1,3,5,6,7,8,9,13};
    vector<int> drop = {2}; // drop fnlwgt column

    if (default_params) {
        ModelParams params = create_default_params();
        params.num_idx = numerical;
        params.cat_idx = categorical;
        parameters.push_back(params);
    } else {
        parameters.back().num_idx = numerical;
        parameters.back().cat_idx = categorical;
    }

    size_t current_index = 0;

    std::string line;
    while (getline(train_infile, line,'\n') && current_index < num_samples) {
        
        if (line.find('?') < line.length()) {
            continue; // drop rows with missing values
        }

        stringstream ss(line);
        vector<string> strings = split_string(line, ',');
        vector<double> X_row;

        // fill X and y
        for(int i=0; i<strings.size()-1; i++){
            if (std::find(drop.begin(), drop.end(), i) != drop.end()) {
                // drop column
                continue;
            } else if (std::find(numerical.begin(), numerical.end(), i) != numerical.end()) {
                // numerical feature
                X_row.push_back(stof(strings[i]));
            { else {
                // categorical feature

                // TODO map with alrady seen labels to numbers
            }
        }
        strings.back().find('>') ? y.push_back(1) : y.push_back(0);
        X.push_back(X_row);
        current_index++;
    }
    
    DataSet dataset = DataSet(X,y);
    dataset.name = num_samples == 300 ? "abalone_small" : "abalone_full";
    return dataset;
}
