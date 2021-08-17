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
#include "dp-gbdt/include/data.h"


// global variables, the following methods store to them:
//  - ecall_load_dataset_into_enclave
//  - ecall_load_modelparams_into_enclave
DataSet *dataset;
ModelParams modelparams; 


/* 
 * sgx_printf: 
 *   Invokes OCALL to display the enclave buffer to the terminal.
 *   At the moment our only way to get output out of the enclave.
 */
void sgx_printf(const char *fmt, ...)
{
    char buf[BUFSIZ] = {'\0'};
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
}


// creates a C++ matrix/vector from the received C-style X and y.
// saves it to global variable "dataset" inside the enclave.
void ecall_load_dataset_into_enclave(sgx_dataset *dset)
{
    VVD X;
    std::vector<double> y;
    for(size_t row=0; row<dset->num_rows; row++){
        std::vector<double> X_row;
        for(size_t col=0; col<dset->num_cols; col++){
            double value = dset->X[row * dset->num_cols + col];
            X_row.push_back(value);
        }
        X.push_back(X_row);
        y.push_back(dset->y[row]);
    }
    dataset = new DataSet(X, y);
    dataset->name = dset->name;
}


// creates C++ ModelParams from the received C-style sgx_modelparams.
// saves it to global variable "modelparams" inside the enclave.
void ecall_load_modelparams_into_enclave(sgx_modelparams *mparams)
{
    modelparams.nb_trees = mparams->nb_trees;
    modelparams.privacy_budget = mparams->privacy_budget;
    // 1 means true/enable
    modelparams.use_dp = mparams->use_dp == 1;
    modelparams.gradient_filtering = mparams->gradient_filtering == 1;
    modelparams.balance_partition = mparams->balance_partition == 1;
    modelparams.leaf_clipping = mparams->leaf_clipping == 1;
    modelparams.scale_y = mparams->scale_y == 1;
    // fill num_idx and cat_idx
    std::vector<int> m_idx;
    for(unsigned i=0; i<mparams->num_idx_len; i++){
        m_idx.push_back(mparams->num_idx[i]);
    }
    modelparams.num_idx = m_idx;
    m_idx = {};
    for(unsigned i=0; i<mparams->cat_idx_len; i++){
        m_idx.push_back(mparams->cat_idx[i]);
    }
    modelparams.cat_idx = m_idx;
    // regression or calssification
    if(std::string(mparams->task).compare(std::string("regression")) == 0){
        modelparams.task = std::shared_ptr<Regression>(new Regression());
    } else {
        modelparams.task = std::shared_ptr<BinaryClassification>(new BinaryClassification());
    }
}


void ecall_start_gbdt()
{
    sgx_printf("Hello from the other side\n");

    sgx_printf("%s_size_%i\n", dataset->name.c_str(), dataset->length);

    // create cross validation inputs
    std::vector<TrainTestSplit *> cv_inputs = create_cross_validation_inputs(dataset, 5);
    delete dataset;

    // do cross validation
    std::vector<double> scores;
    for (auto split : cv_inputs) {

        if(modelparams.scale_y){
            split.train.scale(modelparams, -1, 1);
        }

        // train ensemble
        DPEnsemble ensemble = DPEnsemble(&modelparams);
        ensemble.train(&split.train);
        
        // predict with the test set
        std::vector<double> y_pred = ensemble.predict(split.test.X);

        if(modelparams.scale_y) {
            inverse_scale(modelparams, split.train.scaler, y_pred);
        }

        // compute score
        double score = modelparams.task->compute_score(split.test.y, y_pred);

        scores.push_back(score);
        sgx_printf("%f\n", score);
    }
    for(auto elem : scores){
        sgx_printf("%f ", elem);
    } sgx_printf("\n");
}
