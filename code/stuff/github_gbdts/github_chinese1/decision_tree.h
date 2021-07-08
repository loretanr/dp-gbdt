#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include<iostream>
#include<vector>
#include<cmath>
#include<unordered_set>

#include "data.h"

class SplitResults
{
    public:
    int feature;
    float value;
};

class SplitData
{
    public:
    FeaturesLabels dataset_1;
    FeaturesLabels dataset_2;
};

class DecisionTree
{
    public:
    std::shared_ptr<DecisionTree> left;
    std::shared_ptr<DecisionTree> right;

    int height;
    int maximum_height;
    bool is_leaf;
    int split_feature;
    float split_value;
    float threshold;

    DecisionTree(
                 int h = 1,
                 int max_h = 5,
                 bool is_leaf = false,
                 int feature = -1,
                 float value = 0,
                 float t = 1
                 ):
                 height(h), maximum_height(max_h), is_leaf(is_leaf), split_feature(feature), split_value(value), threshold(t) {};
    
    void build_tree(std::shared_ptr<DecisionTree> tree, FeaturesLabels& dataset);

    SplitResults choose_best_feature(std::shared_ptr<DecisionTree> tree, Data& features, std::vector<float>& labels);

    SplitData split_dataset(Data& features, std::vector<float>& labels, int f_index, float value);

    float compute_mean(std::vector<float>& labels);

    float compute_loss(Data& features, std::vector<float>& labels);

    float predict(std::shared_ptr<DecisionTree> tree, std::vector<float>& data);
};

#endif