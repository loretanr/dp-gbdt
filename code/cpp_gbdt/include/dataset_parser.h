#ifndef PARSER_H
#define PARSER_H

#include "parameters.h"
#include "data.h"
#include "utils.h"


class Parser
{
private:
    std::vector<std::string> split_string(const std::string &s, char delim);

public:
    Parser() {};
    ~Parser() {};

    DataSet get_abalone(std::vector<ModelParams> &parameters, size_t num_samples,
        bool use_default_params = false);
    DataSet get_YearPredictionMSD(std::vector<ModelParams> &parameters,
        size_t num_samples, bool use_default_params = false);
    DataSet get_adult(std::vector<ModelParams> &parameters, size_t num_samples,
        bool use_default_params = false);
};

#endif // PARSER_H