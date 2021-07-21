#ifndef PARSER_H
#define PARSER_H

#include "utils.h"

class Parser
{
private:

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