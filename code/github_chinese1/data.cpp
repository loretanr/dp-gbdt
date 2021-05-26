#include "data.h"

Data* LoadData(const char* input_file)
{
    Data* data = new Data;
    std::string tmp_line;
    std::ifstream inputs;
    std::vector<float> row_data;
    float val;
    inputs.open(input_file); // 这里的形参input_file必须是指针
    while (!inputs.eof())
    {
        getline(inputs, tmp_line, '\n');
        if (tmp_line == "\0")
        {
            return data;
        }
        std::stringstream input_line(tmp_line);
        while (input_line >> val)
        {
            row_data.push_back(val);
        }
        data -> push_back(row_data);
        row_data.clear();
    }
    inputs.close();
    return data;
}

FeaturesLabels split_features_labels(Data* data)
{
    FeaturesLabels features_labels;
    std::vector<float> tmp_vec((*data)[0].size() - 1);
    for (size_t i = 0; i < (*data).size(); ++i)
    {
        tmp_vec.assign((*data)[i].begin(), (*data)[i].end() - 1);
        features_labels.features.push_back(tmp_vec);
        features_labels.labels.push_back((*data)[i].back());
        tmp_vec.clear();
    }
    delete data;
    return features_labels;
}