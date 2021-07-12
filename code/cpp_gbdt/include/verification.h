#ifndef VERIFICATION_H
#define VERIFICATION_H

#include "utils.h"
#include "dp_tree.h"
#include "dp_ensemble.h"
#include "dataset_parser.h"

extern bool VERIFICATION_MODE;
extern bool RANDOMIZATION;

namespace Verification
{
    int main(int argc, char *argv[]);
}

#endif // VERIFICATION_H