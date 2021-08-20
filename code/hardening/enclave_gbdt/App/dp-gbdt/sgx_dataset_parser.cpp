#include "sgx_dataset_parser.h"
#include <memory>
#include <map>
#include <numeric>
#include <sstream>
#include <fstream>
#include <cstring>
#include <string>
#include <algorithm>

/* Parsing:
    The SGX interface (see Enclave.edl) does not allow us to pass std::vectors or 
    other convenient C++ stuff into the enclave via ecalls. Therefore we must 
    parse the datasets into C-style arrays/types. Those can be brought into the 
    enclave. In the enclave we reassemble our std::vectors etc. for GBDT.
    Restrictions:
        - the dataset file needs to be comma separated
        - the dataset must not contain missing values
*/


/* you can add new datasets here */

sgx_dataset SGX_Parser::get_abalone(sgx_modelparams &parameters, int num_samples)
{
    const char *file = "datasets/real/abalone.data";
    const char *name = "abalone";
    int num_rows = 4177;
    int num_cols = 9;
    // std::shared_ptr<Regression> task(new Regression());
    const char *task = "regression";
    std::vector<int> num_idx = {1,2,3,4,5,6,7};
    std::vector<int> cat_idx = {0};
    int target_idx = 8;
    std::vector<int> drop_idx = {};

    return parse_file(file, name, num_rows, num_cols, num_samples, task, num_idx,
        cat_idx, target_idx, drop_idx, parameters);
}


/* helper functions */

std::vector<std::string> SGX_Parser::split_string(const std::string &s, char delim)
{
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        result.push_back(item);
    }
    return result;
}

bool SGX_Parser::find_elem(int elem, int array[], int size)
{
    for( int i=0 ; i < size; i++){
        if(array[i] == elem){
            return true; 
        }
    }
    return false; 
}

sgx_dataset SGX_Parser::parse_file(const char *dataset_file, const char *dataset_name, int num_rows,
        int num_cols, int num_samples, const char *task, std::vector<int> num_idx,
        std::vector<int> cat_idx, int target_idx, std::vector<int> drop_idx,
        sgx_modelparams &modelparams)
{
    std::ifstream infile(dataset_file);
    std::string line;
    num_samples = std::min(num_samples, num_rows);

    modelparams.num_idx = (int *) malloc(num_idx.size() * sizeof(int));
    for(size_t i=0; i<num_idx.size(); i++){
        modelparams.num_idx[i] = num_idx[i];
    }
    modelparams.cat_idx = (int *) malloc(cat_idx.size() * sizeof(int));
    for(size_t i=0; i<cat_idx.size(); i++){
        modelparams.cat_idx[i] = cat_idx[i];
    }
    modelparams.task = (char *) malloc(std::strlen(task) * sizeof(char));
    for(size_t i=0; i< strlen(task); i++){
        modelparams.task[i] = task[i];
    }
    modelparams.num_idx_len = (unsigned) num_idx.size();
    modelparams.cat_idx_len = (unsigned) cat_idx.size();
    modelparams.task_len = (unsigned) std::strlen(task);

    int num_used_cols = num_cols - (int) drop_idx.size() - 1;
    double *X = (double *) malloc(num_used_cols * num_samples * sizeof(double));
    double *y = (double *) malloc(num_samples * sizeof(double));

    // parse dataset, label-encode categorical features
    int current_index = 0;
    std::vector<std::map<std::string,double>> mappings(num_used_cols + 1); // last (additional) one is for y

    while (std::getline(infile, line,'\n') && current_index < num_samples) {
        std::stringstream ss(line);
        std::vector<std::string> strings = split_string(line, ',');

        // go through each column
        int current_used_col = 0;
        for(int i=0; i< (int) strings.size(); i++){

            // is it a drop column?
            if (std::find(drop_idx.begin(), drop_idx.end(), i) != drop_idx.end()) {
                continue;
            }

            // y
            if(i == target_idx){
                if (std::string(task).compare(std::string("regression")) == 0) {
                    // regression -> y is numerical
                    y[current_index] = stof(strings[i]);
                } else {
                    try { // categorical
                        double dummy_value = mappings.back().at(strings[i]);
                        y[current_index] = dummy_value;
                    } catch (const std::out_of_range& oor) {
                        // new label encountered, create mapping
                        mappings.back().insert({strings[i], mappings.back().size()});
                        double dummy_value = mappings.back().at(strings[i]);
                        y[current_index] = dummy_value;
                    }
                }
                continue;
            }

            
            // X
            if (std::find(num_idx.begin(), num_idx.end(), i) != num_idx.end()) {
                // numerical feature
                X[current_index * num_used_cols + current_used_col++] = std::stof(strings[i]);
            } else {
                // categorical feature, do label-encoding
                try {
                    double dummy_value = mappings[i].at(strings[i]);
                    X[current_index * num_used_cols + current_used_col] = dummy_value;
                } catch (const std::out_of_range& oor) {
                    // new label encountered, create mapping
                    mappings[i].insert({strings[i], mappings[i].size()});
                    double dummy_value = mappings[i].at(strings[i]);
                    X[current_index * num_used_cols + current_used_col] = dummy_value;
                }
                current_used_col++;
            }
        }
        current_index++;
    }
    sgx_dataset dataset;
    dataset.X = X;
    dataset.y = y;
    dataset.num_rows = (unsigned) num_samples;
    dataset.num_cols = (unsigned) num_used_cols;
    dataset.name = new char[strlen(dataset_name)];
    strcpy(dataset.name, dataset_name);
    //  = (char *) (std::string(dataset_name) + std::string("_size_")); // + std::string(itoa(num_samples));// + (char *) itoa(num_samples);
    return dataset;
}
