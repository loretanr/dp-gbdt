#include <memory>
#include <map>
#include <numeric>
#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>
#include "dataset_parser.h"


DataSet Parser::get_abalone(std::vector<ModelParams> &parameters, size_t num_samples, bool use_default_params)
{
    std::ifstream infile("datasets/real/abalone.data");
    std::string line; VVD X; std::vector<double> y;
    num_samples = std::min(num_samples, (size_t) 4177);

    // regression task -> LSE
    std::shared_ptr<LeastSquaresError> lossfunction(new LeastSquaresError());

    if (use_default_params) {
        // create some default parameters
        ModelParams params = create_default_params();
        params.lossfunction = lossfunction;
        params.cat_idx = {0}; // first column is categorical
        params.num_idx = {1,2,3,4,5,6,7};
        parameters.push_back(params);
    } else {
        // you have already defined your parameters, then just abalone specific ones are added
        parameters.back().num_idx = {1,2,3,4,5,6,7};
        parameters.back().cat_idx = {0};
        parameters.back().lossfunction = lossfunction;
    }

    // parse dataset, label-encode categorical feature
    size_t current_index = 0;
    while (std::getline(infile, line,'\n') && current_index < num_samples) {
        std::stringstream ss(line);
        std::vector<std::string> strings = split_string(line, ',');
        std::vector<double> X_row;
        if(strings[0] == "M"){ // first col categorical (gender M/F/I)
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
    switch(num_samples){
        case 300: dataset.name = "abalone_small"; break;
        case 4177: dataset.name = "abalone_full"; break;
        default: dataset.name = std::string("abalone_custom_size_").append(
            std::to_string(num_samples));
    }
    return dataset;
}


DataSet Parser::get_YearPredictionMSD(std::vector<ModelParams> &parameters, size_t num_samples, bool use_default_params)
{
    std::ifstream infile("datasets/real/YearPredictionMSD.txt");
    std::string line; VVD X; std::vector<double> y;

    // regression task -> LSE
    std::shared_ptr<LeastSquaresError> lossfunction(new LeastSquaresError());

    // all 90 columns are numerical -> create vector with numbers 0..89
    std::vector<int> num_idx(90);
    std::iota(std::begin(num_idx), std::end(num_idx), 0);
    if (use_default_params){
        // create some default parameters
        ModelParams params = create_default_params();
        params.cat_idx = {};
        params.num_idx = num_idx;
        params.lossfunction = lossfunction;
        parameters.push_back(params);
    } else {
        // you have already defined your parameters, then just yearMSD specific ones are added
        parameters.back().cat_idx = {};
        parameters.back().num_idx = num_idx;
        parameters.back().lossfunction = lossfunction;
    }

    // parse dataset, only has numerical features
    size_t current_index = 0;
    while (std::getline(infile, line,'\n') && current_index < num_samples) {
        std::stringstream ss(line);
        std::vector<std::string> strings = split_string(line, ',');
        std::vector<double> X_row;
        for(size_t i=1;i<strings.size(); i++){
            X_row.push_back(stof(strings[i]));
        }
        y.push_back(stof(strings[0]));
        X.push_back(X_row);
        current_index++;
    }

    DataSet dataset = DataSet(X,y);
    switch(num_samples){
        case 300: dataset.name = "yearMSD_small"; break;
        case 1000: dataset.name = "yearMSD_medium"; break;
        default: dataset.name = std::string("yearMSD_custom_size_").append(
            std::to_string(num_samples));
    }
    return dataset;
}


DataSet Parser::get_adult(std::vector<ModelParams> &parameters, size_t num_samples, bool use_default_params)
{
    std::ifstream train_infile("datasets/real/adult.data");
    std::ifstream test_infile("datasets/real/adult.test");
    VVD X; std::vector<double> y;

    // column types for parsing
    std::vector<int> numerical = {0,4,10,11,12};
    std::vector<int> categorical = {1,3,5,6,7,8,9,13};
    std::vector<int> drop = {2}; // drop fnlwgt column

    int nb_trees = 5;           // TODO remove, for debug
    // if (num_samples <= 300)
    //     nb_trees = 5;


    // create / adjust model parameters
    std::shared_ptr<BinomialDeviance> lossfunction(new BinomialDeviance());
    if (use_default_params) {
        ModelParams params = create_default_params();
        params.num_idx = {0,3,9,10,11}; // adjusted for dropped column
        params.cat_idx = {1,2,4,5,6,7,8,12};
        params.lossfunction = lossfunction;
        params.nb_trees = nb_trees;
        parameters.push_back(params);
    } else {
        parameters.back().num_idx = {0,3,9,10,11};
        parameters.back().cat_idx = {1,2,4,5,6,7,8,12};
        parameters.back().lossfunction = lossfunction;
        parameters.back().nb_trees = nb_trees;
    }
    
    // use this map for label encoding of categorical features (string -> float)
    std::vector<std::map<std::string,float>> mappings(14);

    // parse training dataset
    size_t current_index = 0;
    std::string line;
    while (std::getline(train_infile, line,'\n') && current_index < num_samples) {
        
        // drop dataset rows that contain missing entries
        if (line.find('?') < line.length() or line.empty()) {
            continue;
        }

        // fill X and y
        std::stringstream ss(line);
        std::vector<std::string> strings = split_string(line, ',');
        std::vector<double> X_row;
        for(size_t i=0; i<strings.size()-1; i++){
            if (std::find(drop.begin(), drop.end(), i) != drop.end()) {
                // drop column
                continue;
            } else if (std::find(numerical.begin(), numerical.end(), i) != numerical.end()) {
                // numerical feature
                X_row.push_back(stof(strings[i]));
            } else {
                // categorical feature, do label-encoding
                try {
                    float dummy_value = mappings[i].at(strings[i]);
                    X_row.push_back(dummy_value);
                } catch (const std::out_of_range& oor) {
                    // new label encountered, create mapping
                    mappings[i].insert({strings[i], mappings[i].size()});
                    float dummy_value = mappings[i].at(strings[i]);
                    X_row.push_back(dummy_value);
                }
            }
        }

        // y=1 for ">50k", y=0 for "<=50k"
        (strings.back().find('>') < strings.back().length()) ? y.push_back(1) : y.push_back(0);
        X.push_back(X_row);
        current_index++;
    }

    // TODO
    // go through all lines of the test dataset file and append to X and y (we'll have our own splits)
    // for now it's fine since we're using small amounts of samples (training is already 30k)


    DataSet dataset = DataSet(X,y);
    switch(num_samples){
        case 300: dataset.name = "adult_small"; break;
        case 1000: dataset.name = "adult_medium"; break;
        default: dataset.name = std::string("adult_custom_size_").append(
            std::to_string(num_samples));
    }
    dataset.task = "classification";  // TODO unsused for now
    return dataset;
}
