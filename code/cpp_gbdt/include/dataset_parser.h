#ifndef PARSER_H
#define PARSER_H

#include "utils.h"


class Parser
{
private:

public:
    Parser() {};
    ~Parser() {};

    DataSet get_abalone(ModelParams &params);
};

#endif // PARSER_H