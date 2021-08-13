/*
 * Copyright (C) 2011-2020 Intel Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of Intel Corporation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <stdarg.h>
#include <stdio.h>      /* vsnprintf */

#include "Enclave.h"
#include "Enclave_t.h"  /* print_string */

#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "parameters.h"
#include "dp-gbdt/include/dp_ensemble.h"
#include "dp-gbdt/include/dataset_parser.h"
#include "dp-gbdt/include/data.h"

/* 
 * printf: 
 *   Invokes OCALL to display the enclave buffer to the terminal.
 */
void printf(const char *fmt, ...)
{
    char buf[BUFSIZ] = {'\0'};
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
}

gaggi global_dataset;

void ecall_pass_in_dataset(gaggi dataset)
{
    global_dataset = dataset;
}


/* Where the real deal starts */

void ecall_start_gbdt(int testnumber)
{
    printf("I'm alive\n");
    // Define model parameters
    // reason to use a vector is because parser expects it
    std::vector<ModelParams> parameters;
    ModelParams current_params = create_default_params();

    // change model params here if required:
    current_params.privacy_budget = 5;
    current_params.nb_trees = 10;
    current_params.use_dp = false;
    current_params.gradient_filtering = true;
    current_params.balance_partition = true;
    current_params.leaf_clipping = true;
    current_params.scale_y = false;
    parameters.push_back(current_params);

    // Choose your dataset
    DataSet dataset; // = Parser::get_abalone(parameters, 5000, false);

    printf("%s\n", dataset.name);

    // create cross validation inputs
    std::vector<TrainTestSplit> cv_inputs = create_cross_validation_inputs(dataset, 5);

    // do cross validation
    std::vector<double> rmses;
    for (auto split : cv_inputs) {
        ModelParams params = parameters[0];

        if(params.scale_y){
            split.train.scale(params, -1, 1);
        }

        DPEnsemble ensemble = DPEnsemble(&params);
        ensemble.train(&split.train);
        
        // predict with the test set
        std::vector<double> y_pred = ensemble.predict(split.test.X);

        if(params.scale_y) {
            inverse_scale(params, split.train.scaler, y_pred);
        }

        // compute score
        double score = params.task->compute_score(split.test.y, y_pred);

        printf("%d ", score);
    } printf("\n");
}
