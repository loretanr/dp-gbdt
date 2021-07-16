#ifndef PARSER_H
#define PARSER_H

#include "utils.h"


class Parser
{
private:

public:
    Parser() {};
    ~Parser() {};

    DataSet get_abalone(vector<ModelParams> &parameters, size_t num_samples = false, bool default_params = false);
    DataSet get_YearPredictionMSD(vector<ModelParams> &parameters, size_t num_samples = false, bool default_params = false);
    DataSet get_adult(vector<ModelParams> &parameters, size_t num_samples = false, bool default_params = false);
};

#endif // PARSER_H