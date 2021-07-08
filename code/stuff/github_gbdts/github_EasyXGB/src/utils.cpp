#include "utils.h"

void Timer::start() {
    time_start = std::chrono::high_resolution_clock::now();
}

void Timer::stop() {
    time_stop = std::chrono::high_resolution_clock::now();
}

double Timer::count() {
    auto span = std::chrono::duration_cast<std::chrono::duration<double>>(time_stop - time_start);
    return span.count();
}

int read_data_from_csv(string file_path, const vector<int>& attribute_types, string objective,
                       vector<vector<float>>& data, vector<float>& labels,
                       map<int, map<string, int>>& categoric_attr_maps,
                       map<int, map<int, string>>& categoric_attr_inverse_maps,
                       map<string, int>& label_maps, map<int, string>& label_inverse_maps) {
    ifstream ifs(file_path);
    string tmp_line;
    while (getline(ifs, tmp_line)) {
        vector<float> nums = parse_csv_line(tmp_line, attribute_types, objective,
                                            categoric_attr_maps, categoric_attr_inverse_maps,
                                            label_maps, label_inverse_maps);
        labels.push_back(nums.back());
        nums.pop_back();
        data.push_back(nums);
    }
    if (check_data_valid(data, labels)) {
        return 1;
    }
    return 0;
}

vector<float> parse_csv_line(const string& line, const vector<int>& attribute_types, string objective,
                             map<int, map<string, int>>& categoric_attr_maps,
                             map<int, map<int, string>>& categoric_attr_inverse_maps,
                             map<string, int>& label_maps, map<int, string>& label_inverse_maps) {
    stringstream ss(line);
    vector<float> nums;
    string tmp;
    int index = 0;
    while (getline(ss, tmp, ',')) {
        if (index > attribute_types.size()) {
            cout << "data invalid, check data...";
            cout << "num attributes(" << index << ") != "
                 << "num attribute types(" << attribute_types.size() << ")" << endl;
            exit(1);
        }
        if (index < attribute_types.size()) {
            if (attribute_types[index] == 2) {
                if (categoric_attr_maps[index].find(tmp) == categoric_attr_maps[index].end()) {
                    categoric_attr_maps[index][tmp] = categoric_attr_maps[index].size();
                    categoric_attr_inverse_maps[index][categoric_attr_maps[index][tmp]] = tmp;
                }
                nums.push_back(categoric_attr_maps[index][tmp]);
            } else if (attribute_types[index] == 1) {
                if (tmp == "") {
                    nums.push_back(0);
                } else {
                    nums.push_back(stof(tmp));
                }
            } else {
                cout << "attribute type invalid, check type..." << endl;
                exit(1);
            }
        } else {
            if (objective == "regression") {
                nums.push_back(stof(tmp));

            } else if (objective == "binary") {
                tmp = to_string(stoi(tmp));
                if (label_maps.find(tmp) == label_maps.end()) {
                    label_maps[tmp] = label_maps.size() - 1;
                    label_inverse_maps[label_maps[tmp]] = tmp;
                }
                if (label_maps.size() > 2) {
                    cout << "Currently only support binary classification...";
                    exit(1);
                }
                nums.push_back(label_maps[tmp]);
            }
        }
        index++;
    }
    return nums;
}

int check_data_valid(const vector<vector<float>>& data, const vector<float>& labels) {
    if (data.size() == 0 || labels.size() == 0 || data[0].size() == 0) {
        return 1;
    }
    if (data.size() != labels.size()) {
        return 1;
    }
    unsigned num_attributes = data[0].size();
    for (unsigned i = 1; i < data.size(); ++i) {
        if (data[i].size() != num_attributes) {
            return 1;
        }
    }
    return 0;
}

void shuffle_data_set(vector<vector<float>>& data, vector<float>& labels) {
    shuffle(data.begin(), data.end(), default_random_engine(1));
    shuffle(labels.begin(), labels.end(), default_random_engine(1));
}

void shuffle_attributes(vector<int>& attributes) {
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(attributes.begin(), attributes.end(), default_random_engine(seed));
}

float square_first_gradient(float y, float y_pred) {
    return 2 * (y_pred - y);
}

float square_second_gradient(float y, float y_pred) {
    return 2.0;
}

float logistic_first_gradient(float y, float y_pred) {
    float tmp1 = exp(y_pred) + 1;
    float tmp2 = exp(y_pred) + exp(-y_pred) + 2;
    return tmp1 / tmp2 - y;
}

float logistic_second_gradient(float y, float y_pred) {
    float tmp = exp(y_pred) + exp(-y_pred) + 2;
    return 1 / tmp;
}
