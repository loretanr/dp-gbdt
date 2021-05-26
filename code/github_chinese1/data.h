#ifndef LoadData_H
#define LoadData_H

#include<iostream>
#include<vector>
#include<fstream>
#include<sstream>

typedef std::vector<std::vector<float>> Data;

class FeaturesLabels
{
    public:
    Data features;
    std::vector<float> labels;
    size_t size = features.size();
};

Data* LoadData(const char* input_file);

FeaturesLabels split_features_labels(Data* data);

#endif