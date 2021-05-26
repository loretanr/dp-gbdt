#include <cmath>
#include <iostream>
#include "Dataset.h"

Dataset::Dataset(string file_path, vector<int> attribute_types,
                 string objective, float test_ratio):
    attribute_types(attribute_types), test_ratio(test_ratio) {
    if (read_data_from_csv(file_path, attribute_types, objective, data, labels,
                           categoric_attr_maps, categoric_attr_inverse_maps,
                           label_maps, label_inverse_maps)) {
        cout << "data invalid, check data...";
        exit(1);
    }
    if (attribute_types.size() != num_attributes()) {
        cout << "data invalid, check data...";
        cout << "num attributes(" << num_attributes() << ") != "
             << "num attribute types(" << attribute_types.size() << ")" << endl;
        exit(1);
    }
    create_data_set();
    cout << "read data from " << file_path << " done..." << endl;
    cout << "num instances: " << num_instances() << endl;
    cout << "num attributes: " << num_attributes() << endl;
    int num_positive = accumulate(labels.begin(), labels.end(), 0);
    cout << "num positive: " << num_positive << endl;
    cout << "num negative: " << num_instances() - num_positive << endl;
}

void Dataset::create_data_set() {
    shuffle_data_set(data, labels);
    preds = vector<float>(labels.size(), 0);
    int num_test = int(num_instances() * test_ratio);
    for (int i = 0; i < num_instances(); ++i) {
        if (i < num_test) {
            test_set.push_back(i);
        } else {
            train_set.push_back(i);
        }
    }
}

vector<int> Dataset::instances(string set_name) {
    if (set_name == "train") {
        return train_set;
    } else if (set_name == "valid") {
        return valid_set;
    } else if (set_name == "test") {
        return test_set;
    } else {
        return train_set;
    }
}

vector<float>& Dataset::get(int index) {
    return data[index];
}

float Dataset::get(int index, int attribute) {
    return data[index][attribute];
}

float Dataset::get_label(int index) {
    return labels[index];
}

float Dataset::get_pred(int index) {
    return preds[index];
}

int Dataset::get_attr_type(int index) {
    return attribute_types[index];
}

void Dataset::reset_preds() {
    preds = vector<float>(labels.size(), 0);
}

void Dataset::set_pred(int index, float pred) {
    preds[index] += pred;
}

int Dataset::num_instances() {
    return data.size();
}

int Dataset::num_attributes() {
    return data[0].size();
}

float Dataset::rmse(string set_name) {
    vector<int>* data_set;
    if (set_name == "train") {
        data_set = &train_set;
    } else if (set_name == "valid") {
        data_set = &valid_set;
    } else if (set_name == "test") {
        data_set = &test_set;
    } else {
        data_set = &train_set;
    }

    float square_sum = 0.0;
    for (int i: *data_set) {
        square_sum += pow(labels[i]-preds[i], 2);
    }
    if (data_set->size() == 0) {
        return 0;
    }
    return sqrt(square_sum / data_set->size());
}

float Dataset::acc(string set_name) {
    vector<int>* data_set;
    if (set_name == "train") {
        data_set = &train_set;
    } else if (set_name == "valid") {
        data_set = &valid_set;
    } else if (set_name == "test") {
        data_set = &test_set;
    } else {
        data_set = &train_set;
    }

    int correct = 0;
    for (int i: *data_set) {
        if (labels[i] == max(0, min(1, int(preds[i]+0.5)))) {
            correct++;
        }
    }
    if (data_set->size() == 0) {
        return 0;
    }
    return float(correct) / data_set->size();
}

void Dataset::switch_new_fold(int fold_index, int num_folds) {
    train_set.clear();
    valid_set.clear();
    fold_index %= num_folds;

    int num_test = int(num_instances() * test_ratio);
    int num_train_valid = num_instances() - num_test;
    int num_instances_per_fold = num_train_valid / num_folds;

    if (num_folds == 1) {
        for (int i = num_test; i < num_instances(); ++i) {
            train_set.push_back(i);
        }
        return;
    }

    int train_start = num_test;
    int train_end = num_test + fold_index * num_instances_per_fold;
    for (int i = train_start; i < train_end; ++i) {
        train_set.push_back(i);
    }
    train_start = num_test + (fold_index + 1) * num_instances_per_fold;
    train_end = num_instances();
    for (int i = train_start; i < train_end; ++i) {
        train_set.push_back(i);
    }

    int valid_start = num_test + fold_index * num_instances_per_fold;
    int valid_end = num_test + (fold_index + 1) * num_instances_per_fold;
    for (int i = valid_start; i < valid_end; ++i) {
        valid_set.push_back(i);
    }
}

