#ifndef PARSER_H
#define PARSER_H

#include "parameters.h"
#include "data.h"
#include "utils.h"


class Parser
{
private:
    // methods
    static std::vector<std::string> split_string(const std::string &s, char delim);
    static DataSet *parse_file(std::string dataset_file, std::string dataset_name, int num_rows, int num_cols, int num_samples, 
        std::shared_ptr<Task> task, std::vector<int> num_idx, std::vector<int> cat_idx, std::vector<int> target_idx, 
        std::vector<int> drop_idx, std::vector<ModelParams> &parameters,bool use_default_params);

public:
    // methods
    static DataSet *get_abalone(std::vector<ModelParams> &parameters, size_t num_samples,
        bool use_default_params = false);
    static DataSet *get_YearPredictionMSD(std::vector<ModelParams> &parameters,
        size_t num_samples, bool use_default_params = false);
    static DataSet *get_adult(std::vector<ModelParams> &parameters, size_t num_samples,
        bool use_default_params = false);
};

#endif // PARSER_H