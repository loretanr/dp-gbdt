#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <map>
#include <cmath>
using namespace std;

struct ModelParam {
    int num_trees = 50;
    int max_depth = 20;
    int min_examples_leaf = 50;
    int num_folds = 10;
    float gamma = 0.0;
    float lambda = 1.0;
    float learning_rate = 0.5;
    float col_sample = 1.0;
    bool print_tree = false;
    bool prune_tree = true;
    bool multi_threads = true;
    string objective = "regression"; // or binary
};

struct Timer {
    std::chrono::high_resolution_clock::time_point time_start, time_stop;
    void start();
    void stop();
    double count();
};

int read_data_from_csv(string file_path, const vector<int>& attribute_types, string objective,
                       vector<vector<float>>& data, vector<float>& labels,
                       map<int, map<string, int>>& categoric_attr_maps,
                       map<int, map<int, string>>& categoric_attr_inverse_maps,
                       map<string, int>& label_maps, map<int, string>& label_inverse_maps);
vector<float> parse_csv_line(const string& line, const vector<int>& attribute_types, string objective,
                             map<int, map<string, int>>& categoric_attr_maps,
                             map<int, map<int, string>>& categoric_attr_inverse_maps,
                             map<string, int>& label_maps, map<int, string>& label_inverse_maps);
int check_data_valid(const vector<vector<float>>& data, const vector<float>& labels);

void shuffle_data_set(vector<vector<float>>& data, vector<float>& labels);
void shuffle_attributes(vector<int>& attributes);

float square_first_gradient(float y, float y_pred);
float square_second_gradient(float y, float y_pred);
float logistic_first_gradient(float y, float y_pred);
float logistic_second_gradient(float y, float y_pred);

#endif // UTILS_H
