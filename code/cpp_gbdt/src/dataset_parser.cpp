#include <memory>
#include <map>
#include <numeric>
#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>
#include "dataset_parser.h"
#include "data.h"


DataSet Parser::get_abalone2(std::vector<ModelParams> &parameters,
        size_t num_samples, bool use_default_params)
{
    std::string file = "datasets/real/abalone.data";
    std::string name = "abalone";
    int num_rows = 4177;
    int num_cols = 9;
    std::shared_ptr<Regression> task(new Regression());
    std::vector<int> num_idx = {1,2,3,4,5,6,7};
    std::vector<int> cat_idx = {0};
    std::vector<int> target_idx = {8};
    std::vector<int> drop_idx = {};

    return parse_file(file, name, num_rows, num_cols, num_samples, task, num_idx,
        cat_idx, target_idx, drop_idx, parameters, use_default_params);
}


DataSet Parser::get_YearPredictionMSD2(std::vector<ModelParams> &parameters, 
        size_t num_samples, bool use_default_params)
{
    std::string file = "datasets/real/YearPredictionMSD.txt";
    std::string name = "yearMSD";
    int num_rows = 515345;
    int num_cols = 90;
    std::shared_ptr<Regression> task(new Regression());
    std::vector<int> num_idx(89);
    std::iota(std::begin(num_idx)++, std::end(num_idx), 1); // num_idx = {1,...,89}
    std::vector<int> cat_idx = {};
    std::vector<int> target_idx = {0};
    std::vector<int> drop_idx = {};

    return parse_file(file, name, num_rows, num_cols, num_samples, task, num_idx,
        cat_idx, target_idx, drop_idx, parameters, use_default_params);
}


DataSet Parser::get_abalone(std::vector<ModelParams> &parameters,
        size_t num_samples, bool use_default_params)
{
    std::ifstream infile("datasets/real/abalone.data");
    std::string line; VVD X; std::vector<double> y;
    num_samples = std::min(num_samples, (size_t) 4177);

    // regression task -> Least Squares
    std::shared_ptr<Regression> task(new Regression());

    if (use_default_params) {
        // create some default parameters
        ModelParams params = create_default_params();
        params.task = task;
        params.cat_idx = {0}; // first column is categorical
        params.num_idx = {1,2,3,4,5,6,7};
        parameters.push_back(params);
    } else {
        // you have already defined your parameters, then just abalone specific ones are added
        parameters.back().num_idx = {1,2,3,4,5,6,7};
        parameters.back().cat_idx = {0};
        parameters.back().task = task;
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
        default: dataset.name = std::string("abalone_custom_size_").append(
            std::to_string(num_samples));
    }
    return dataset;
}


DataSet Parser::get_YearPredictionMSD(std::vector<ModelParams> &parameters, 
        size_t num_samples, bool use_default_params)
{
    std::ifstream infile("datasets/real/YearPredictionMSD.txt");
    std::string line; VVD X; std::vector<double> y;

    // regression task -> Least Squares
    std::shared_ptr<Regression> task(new Regression());

    // all 90 columns are numerical -> create vector with numbers 0..89
    std::vector<int> num_idx(90);
    std::iota(std::begin(num_idx), std::end(num_idx), 0);
    if (use_default_params){
        // create some default parameters
        ModelParams params = create_default_params();
        params.cat_idx = {};
        params.num_idx = num_idx;
        params.task = task;
        parameters.push_back(params);
    } else {
        // you have already defined your parameters, then just yearMSD specific ones are added
        parameters.back().cat_idx = {};
        parameters.back().num_idx = num_idx;
        parameters.back().task = task;
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

DataSet Parser::get_adult2(std::vector<ModelParams> &parameters,
        size_t num_samples, bool use_default_params)
{
    std::string file = "datasets/real/adult.data";
    std::string name = "adult";
    int num_rows = 48842;
    int num_cols = 90;
    std::shared_ptr<BinaryClassification> task(new BinaryClassification());
    std::vector<int> num_idx = {0,4,10,11,12};
    std::vector<int> cat_idx = {1,3,5,6,7,8,9,13};
    std::vector<int> target_idx = {14};
    std::vector<int> drop_idx = {};

    return parse_file(file, name, num_rows, num_cols, num_samples, task, num_idx,
        cat_idx, target_idx, drop_idx, parameters, use_default_params);
}



/** Utility functions */

std::vector<std::string> Parser::split_string(const std::string &s, char delim)
{
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        result.push_back(item);
    }
    return result;
}


DataSet Parser::parse_file(std::string dataset_file, std::string dataset_name, int num_rows,
        int num_cols, int num_samples, std::shared_ptr<Task> task, std::vector<int> num_idx,
        std::vector<int> cat_idx, std::vector<int> target_idx, std::vector<int> drop_idx,
        std::vector<ModelParams> &parameters, bool use_default_params)
{
    std::ifstream infile(dataset_file);
    std::string line;
    VVD X;
    std::vector<double> y;
    num_samples = std::min(num_samples, num_rows);
    if (use_default_params) {
        // create some default parameters
        ModelParams params = create_default_params();
        params.task = task;
        params.cat_idx = cat_idx;
        params.num_idx = num_idx;
        parameters.push_back(params);
    } else {
        // you have already defined your parameters, then just abalone specific ones are added
        parameters.back().num_idx = num_idx;
        parameters.back().cat_idx = cat_idx;
        parameters.back().task = task;
    }

    // parse dataset, label-encode categorical features
    int current_index = 0;
    std::vector<std::map<std::string,float>> mappings(num_cols + 1);

    while (std::getline(infile, line,'\n') && current_index < num_samples) {
        std::stringstream ss(line);
        std::vector<std::string> strings = split_string(line, ',');
        std::vector<double> X_row;

        // drop dataset rows that contain missing entries
        if (line.find('?') < line.length() or line.empty()) {
            continue;
        }

        // go through each column
        for(size_t i=0; i<strings.size(); i++){

            // drop column
            if (std::find(drop_idx.begin(), drop_idx.end(), i) != drop_idx.end()) {
                continue;
            }

            // y
            if(std::find(target_idx.begin(), target_idx.end(), i) != target_idx.end()){
                if (dynamic_cast<Regression*>(task.get())) {
                    y.push_back(stof(strings[i]));
                } else {
                    try { // categorical
                        float dummy_value = mappings.back().at(strings[i]);
                        y.push_back(dummy_value);
                    } catch (const std::out_of_range& oor) {
                        // new label encountered, create mapping
                        mappings.back().insert({strings[i], mappings.back().size()});
                        float dummy_value = mappings.back().at(strings[i]);
                        y.push_back(dummy_value);
                    }
                }
                continue;
            }

            // X
            if (std::find(num_idx.begin(), num_idx.end(), i) != num_idx.end()) {
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
        X.push_back(X_row);
        current_index++;
    }

    DataSet dataset = DataSet(X,y);
    dataset.name = std::string(dataset_name) + std::string("_size_") + std::to_string(num_samples);
    return dataset;
}
