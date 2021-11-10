#ifndef B826C7B9_7FF2_4138_8D93_857AEE43CD41
#define B826C7B9_7FF2_4138_8D93_857AEE43CD41

#include "user_types.h"
#include <vector>
#include <string>

class SGX_Parser
{
private:
    static std::vector<std::string> split_string(const std::string &s, char delim);
    static bool find_elem(int elem, int array[], int size);
    static sgx_dataset parse_file(const char *dataset_file, const char *dataset_name, int num_rows,
        int num_cols, int num_samples, const char *task, std::vector<int> num_idx,
        std::vector<int> cat_idx, int target_idx, std::vector<int> drop_idx,
        sgx_modelparams &modelparams);

public:
    static sgx_dataset get_abalone(sgx_modelparams &parameters, int num_samples);
    static sgx_dataset get_adult(sgx_modelparams &parameters, int num_samples);
    static sgx_dataset get_year(sgx_modelparams &parameters, int num_samples);

};



#endif /* B826C7B9_7FF2_4138_8D93_857AEE43CD41 */
