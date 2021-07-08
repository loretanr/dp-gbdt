#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <numeric>
#include <map>
#include "utils.h"

using namespace std;

class Dataset {
public:
    Dataset(string file_path, vector<int> attribute_types,
            string objective, float test_ratio=0.0);
    vector<int> instances(string set_name);
    vector<float>& get(int index);
    float get(int index, int attribute);
    float get_label(int index);
    float get_pred(int index);
    int get_attr_type(int index);
    void reset_preds();
    void set_pred(int index, float pred);
    int num_instances();
    int num_attributes();
    float rmse(string set_name);
    float acc(string set_name);
    void switch_new_fold(int fold_index, int num_folds);
    map<int, map<string, int>> categoric_attr_maps;
    map<int, map<int, string>> categoric_attr_inverse_maps;
    map<string, int> label_maps;
    map<int, string> label_inverse_maps;

private:
    void create_data_set();
    vector<vector<float>> data;
    vector<float> labels;
    vector<float> preds;
    vector<int> train_set, valid_set, test_set;
    vector<int> attribute_types; // 1 for numeric, 2 for categoric
    float test_ratio;
};

#endif // DATASET_H
