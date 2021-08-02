#ifndef PARSER_H
#define PARSER_H

#include "parameters.h"
#include "data.h"
#include "utils.h"


class Parser
{
private:
    // methods
    std::vector<std::string> split_string(const std::string &s, char delim);
    DataSet parse_file(std::string dataset_file, std::string dataset_name, int num_rows, int num_cols, int num_samples, 
        std::shared_ptr<Task> task, std::vector<int> num_idx, std::vector<int> cat_idx,
        std::vector<int> target_idx, std::vector<ModelParams> &parameters,bool use_default_params);

public:
    // constructors
    Parser() {};
    ~Parser() {};

    // methods
    DataSet get_abalone(std::vector<ModelParams> &parameters, size_t num_samples,
        bool use_default_params = false);
    DataSet get_YearPredictionMSD(std::vector<ModelParams> &parameters,
        size_t num_samples, bool use_default_params = false);
    DataSet get_adult(std::vector<ModelParams> &parameters, size_t num_samples,
        bool use_default_params = false);


    DataSet get_abalone2(std::vector<ModelParams> &parameters,
        size_t num_samples, bool use_default_params = false);
};

#endif // PARSER_H