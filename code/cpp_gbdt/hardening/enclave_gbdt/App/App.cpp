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

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

# include <unistd.h>
# include <pwd.h>
# define MAX_PATH FILENAME_MAX

#include "sgx_urts.h"
#include "App.h"
#include "Enclave_u.h"

/* Global EID shared by multiple threads */
sgx_enclave_id_t global_eid = 0;

typedef struct _sgx_errlist_t {
    sgx_status_t err;
    const char *msg;
    const char *sug; /* Suggestion */
} sgx_errlist_t;

/* Error code returned by sgx_create_enclave */
static sgx_errlist_t sgx_errlist[] = {
    {
        SGX_ERROR_UNEXPECTED,
        "Unexpected error occurred.",
        NULL
    },
    {
        SGX_ERROR_INVALID_PARAMETER,
        "Invalid parameter.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_MEMORY,
        "Out of memory.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_LOST,
        "Power transition occurred.",
        "Please refer to the sample \"PowerTransition\" for details."
    },
    {
        SGX_ERROR_INVALID_ENCLAVE,
        "Invalid enclave image.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ENCLAVE_ID,
        "Invalid enclave identification.",
        NULL
    },
    {
        SGX_ERROR_INVALID_SIGNATURE,
        "Invalid enclave signature.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_EPC,
        "Out of EPC memory.",
        NULL
    },
    {
        SGX_ERROR_NO_DEVICE,
        "Invalid SGX device.",
        "Please make sure SGX module is enabled in the BIOS, and install SGX driver afterwards."
    },
    {
        SGX_ERROR_MEMORY_MAP_CONFLICT,
        "Memory map conflicted.",
        NULL
    },
    {
        SGX_ERROR_INVALID_METADATA,
        "Invalid enclave metadata.",
        NULL
    },
    {
        SGX_ERROR_DEVICE_BUSY,
        "SGX device was busy.",
        NULL
    },
    {
        SGX_ERROR_INVALID_VERSION,
        "Enclave version was invalid.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ATTRIBUTE,
        "Enclave was not authorized.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_FILE_ACCESS,
        "Can't open enclave file.",
        NULL
    },
    {
        SGX_ERROR_NDEBUG_ENCLAVE,
        "The enclave is signed as product enclave, and can not be created as debuggable enclave.",
        NULL
    },
};

/* Check error conditions for loading enclave */
void print_error_message(sgx_status_t ret)
{
    size_t idx = 0;
    size_t ttl = sizeof sgx_errlist/sizeof sgx_errlist[0];

    for (idx = 0; idx < ttl; idx++) {
        if(ret == sgx_errlist[idx].err) {
            if(NULL != sgx_errlist[idx].sug)
                printf("Info: %s\n", sgx_errlist[idx].sug);
            printf("Error: %s\n", sgx_errlist[idx].msg);
            break;
        }
    }
    
    if (idx == ttl)
        printf("Error: Unexpected error occurred.\n");
}

/* Initialize the enclave:
 *   Call sgx_create_enclave to initialize an enclave instance
 */
int initialize_enclave(void)
{
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    
    /* Call sgx_create_enclave to initialize an enclave instance */
    /* Debug Support: set 2nd parameter to 1 */
    ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, NULL, NULL, &global_eid, NULL);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        return -1;
    }

    return 0;
}

/* OCall functions */
void ocall_print_string(const char *str)
{
    /* Proxy/Bridge will check the length and null-terminate 
     * the input string to prevent buffer overflow. 
     */
    printf("%s", str);
}

/* =================== Parsing ==================== */
#include <memory>
#include <map>
#include <numeric>
#include <sstream>
#include <fstream>
#include <cstring>
#include <string>
#include <algorithm>
#include <vector>

/* Parsing:
    - the data file needs to be comma separated
    - the dataset must not contain missing values!
*/

std::vector<std::string> split_string(const std::string &s, char delim)
{
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        result.push_back(item);
    }
    return result;
}

bool find_elem(int elem, int array[], int size)
{
    for( int i=0 ; i < size; i++){
        if(array[i] == elem){
            return true; 
        }
    }
    return false; 
}

sgx_dataset parse_file(const char *dataset_file, const char *dataset_name, int num_rows,
        int num_cols, int num_samples, const char *task, std::vector<int> num_idx,
        std::vector<int> cat_idx, int target_idx, std::vector<int> drop_idx,
        sgx_modelparams &modelparams)
{
    std::ifstream infile(dataset_file);
    std::string line;
    num_samples = std::min(num_samples, num_rows);

    int fill_index = 0;
    int *num_idx_arr = (int *) malloc(num_idx.size() * sizeof(int));
    for(auto elem : num_idx){
        num_idx_arr[fill_index++] = elem;
    }
    int *cat_idx_arr = (int *) malloc(cat_idx.size() * sizeof(int));
    fill_index = 0;
    for(auto elem : cat_idx){
        cat_idx_arr[fill_index++] = elem;
    }
    char *task_arr = (char *) malloc(std::strlen(task) * sizeof(char));
    for(int i=0; i<strlen(task); i++){
        task_arr[i] = task[i];
    }

    modelparams.num_idx = num_idx_arr;
    modelparams.num_idx_len = num_idx.size();
    modelparams.cat_idx = cat_idx_arr;
    modelparams.cat_idx_len = cat_idx.size();
    modelparams.task = task_arr;
    modelparams.task_len = std::strlen(task);

    int num_used_cols = (int) (num_idx.size() + cat_idx.size() - drop_idx.size());
    double *X = (double *) malloc(num_used_cols * num_samples * sizeof(double));
    double *y = (double *) malloc(num_samples * sizeof(double));

    // parse dataset, label-encode categorical features
    int current_index = 0;
    std::vector<std::map<std::string,double>> mappings(num_used_cols + 1); // last (additional) one is for y

    while (std::getline(infile, line,'\n') && current_index < num_samples) {
        std::stringstream ss(line);
        std::vector<std::string> strings = split_string(line, ',');

        // go through each column
        int current_used_col = 0;
        for(int i=0; i<strings.size(); i++){

            // is it a drop column?
            if (std::find(drop_idx.begin(), drop_idx.end(), i) != drop_idx.end()) {
                continue;
            }

            // y
            if(i == target_idx){
                if (std::string(task).compare(std::string("regression")) == 0) {
                    // regression -> y is numerical
                    y[current_index] = stof(strings[i]);
                } else {
                    try { // categorical
                        double dummy_value = mappings.back().at(strings[i]);
                        y[current_index] = dummy_value;
                    } catch (const std::out_of_range& oor) {
                        // new label encountered, create mapping
                        mappings.back().insert({strings[i], mappings.back().size()});
                        double dummy_value = mappings.back().at(strings[i]);
                        y[current_index] = dummy_value;
                    }
                }
                continue;
            }

            
            // X
            if (std::find(num_idx.begin(), num_idx.end(), i) != num_idx.end()) {
                // numerical feature
                X[current_index * num_used_cols + current_used_col++] = std::stof(strings[i]);
            } else {
                // categorical feature, do label-encoding
                try {
                    double dummy_value = mappings[i].at(strings[i]);
                    X[current_index * num_used_cols + current_used_col] = dummy_value;
                } catch (const std::out_of_range& oor) {
                    // new label encountered, create mapping
                    mappings[i].insert({strings[i], mappings[i].size()});
                    double dummy_value = mappings[i].at(strings[i]);
                    X[current_index * num_used_cols + current_used_col] = dummy_value;
                }
                current_used_col++;
            }
        }
        current_index++;
    }
    sgx_dataset dataset;
    dataset.X = X;
    dataset.y = y;
    dataset.num_rows = num_samples;
    dataset.num_cols = num_used_cols;
    dataset.name = new char[strlen(dataset_name)];
    strcpy(dataset.name, dataset_name);
    //  = (char *) (std::string(dataset_name) + std::string("_size_")); // + std::string(itoa(num_samples));// + (char *) itoa(num_samples);
    return dataset;
}

sgx_dataset get_abalone(sgx_modelparams &parameters, size_t num_samples)
{
    const char *file = "datasets/real/abalone.data";
    const char *name = "abalone";
    int num_rows = 4177;
    int num_cols = 9;
    // std::shared_ptr<Regression> task(new Regression());
    const char *task = "regression";
    std::vector<int> num_idx = {1,2,3,4,5,6,7};
    std::vector<int> cat_idx = {0};
    int target_idx = 8;
    std::vector<int> drop_idx = {};

    return parse_file(file, name, num_rows, num_cols, num_samples, task, num_idx,
        cat_idx, target_idx, drop_idx, parameters);
}

/* ================================================ */



sgx_modelparams create_some_params()
{
    sgx_modelparams params;
    params.use_dp = 1;
    return params;
}


/* Application entry */
int SGX_CDECL main(int argc, char *argv[])
{
    (void)(argc);
    (void)(argv);


    /* Initialize the enclave */
    if(initialize_enclave() < 0){
        printf("Enter a character before exit ...\n");
        getchar();
        return -1; 
    }
 
    /* Utilize trusted libraries */ 
    // ecall_libcxx_functions();
    // double *matrix = (double *) malloc(3 * 3 * sizeof(double));
    // for(int i=0; i<9; i++){ matrix[i] = 42;}

    sgx_modelparams modelparams = create_some_params();
    sgx_dataset dataset = get_abalone(modelparams, 300);

    ecall_load_dataset_into_enclave(global_eid, &dataset);
    ecall_load_modelparams_into_enclave(global_eid, &modelparams);
    
    ecall_start_gbdt(global_eid, 42);
    
    /* Destroy the enclave */
    sgx_destroy_enclave(global_eid);
    
    printf("Info: Cxx11DemoEnclave successfully returned.\n");

    //printf("Enter a character before exit ...\n");
    //getchar();
    return 0;
}

