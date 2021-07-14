#ifndef PARSER_H
#define PARSER_H

#include "utils.h"


class Parser
{
private:

public:
    Parser() {};
    ~Parser() {};

    DataSet get_abalone(ModelParams &params, bool small_subset = false);
    DataSet get_YearPredictionMSD(ModelParams &params, bool small_subset = false);
};

#endif // PARSER_H