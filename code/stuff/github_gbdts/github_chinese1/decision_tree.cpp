#include "data.h"
#include "decision_tree.h"

using namespace std;
    
void DecisionTree::build_tree(shared_ptr<DecisionTree> tree, FeaturesLabels& dataset)
{
    Data features = dataset.features;
    vector<float> labels = dataset.labels;
    if (features.size() == 0)
    {
        return;
    }
    SplitResults feature_and_val = choose_best_feature(tree, features, labels);
    tree -> split_feature = feature_and_val.feature;
    tree -> split_value = feature_and_val.value;
    if (tree -> is_leaf == true || tree -> split_feature == -1)
    {
        return;
    }
    SplitData new_dataset = split_dataset(features, labels, tree -> split_feature, tree -> split_value);
    Data& features_1 = new_dataset.dataset_1.features;
    vector<float> labels_1 = new_dataset.dataset_1.labels;
    Data& features_2 = new_dataset.dataset_2.features;
    vector<float> labels_2 = new_dataset.dataset_2.labels;

    // 此处因为是在类/函数内定义变量，必须要分配动态内存。下面两行是不用智能指针的写法，但new需和delete配合。
    // tree -> left = new DecisionTree(tree -> height + 1);
    // tree -> right = new DecisionTree(tree -> height + 1);

    // 利用智能指针后，需要将类/函数的接口也都统一一下。
    tree -> left = make_shared<DecisionTree>(tree -> height + 1);
    tree -> right = make_shared<DecisionTree>(tree -> height + 1);

    build_tree(tree -> left, new_dataset.dataset_1);
    build_tree(tree -> right, new_dataset.dataset_2);
    return;
}

SplitResults DecisionTree::choose_best_feature(shared_ptr<DecisionTree> tree, Data& features, vector<float>& labels){

    SplitResults result;

    // 先判断下数据集是否已经完全同化
    bool flag = true;
    float value = labels.front();
    for (size_t i=0; i < features.size(); ++i)
    {
        if (labels[i] != value)
        {
            flag = false;
            break;
        }
    }
    if (flag == true || tree -> height >= tree -> maximum_height)
    {
        result.feature = -1;
        result.value = compute_mean(labels);
        tree -> is_leaf = true;
        return result;
    }
    float loss = compute_loss(features, labels);
    float maximum_gain = 0;
    for (size_t i = 0; i < features[0].size(); ++i)
    {
        for (size_t j = 0; j < features.size(); ++j)
        {
            SplitData new_dataset = split_dataset(features, labels, i, features[j][i]);
            Data& features_1 = new_dataset.dataset_1.features;
            vector<float> labels_1 = new_dataset.dataset_1.labels;
            Data& features_2 = new_dataset.dataset_2.features;
            vector<float> labels_2 = new_dataset.dataset_2.labels;
            float new_loss = compute_loss(features_1, labels_1) + compute_loss(features_2, labels_2);
            float gain = loss - new_loss;
            if (gain < tree -> threshold)
            {
                continue;
            }
            if (gain > maximum_gain)
            {
                result.feature = i;
                result.value = features[j][i];
                maximum_gain = gain;
            }
        }
    }
    if (maximum_gain == 0 || maximum_gain < tree -> threshold)
    {
        result.feature = -1;
        result.value = compute_mean(labels);
        tree -> is_leaf = true;
    }
    return result;
}

SplitData DecisionTree::split_dataset(Data& features, vector<float>& labels, int f_index, float value){
    SplitData new_dataset;
    for (size_t i = 0; i < features.size(); ++i)
    {
        if (features[i][f_index] <= value)
        {
            new_dataset.dataset_1.features.push_back(features[i]);
            new_dataset.dataset_1.labels.push_back(labels[i]);
        }
        else
        {
            new_dataset.dataset_2.features.push_back(features[i]);
            new_dataset.dataset_2.labels.push_back(labels[i]);
        }
    }
    return new_dataset;
}

float DecisionTree::compute_mean(vector<float>& labels){
    float total = 0;
    for (size_t i = 0; i < labels.size(); ++i)
    {
        total += labels[i];
    }
    return total / labels.size();
}

float DecisionTree::compute_loss(Data& features, vector<float>& labels){
    float loss = 0;
    float mean = compute_mean(labels);
    for (size_t i = 0; i < features.size(); ++i)
    {
        loss += (labels[i] - mean) * (labels[i] - mean);
    }
    return loss;
}

float DecisionTree::predict(shared_ptr<DecisionTree> tree, vector<float>& example)
{
    if (tree -> is_leaf == true || tree -> split_feature == -1)
    {
        return tree -> split_value;
    }
    if (example[tree -> split_feature] <= tree -> split_value)
    {
        return predict(tree -> left, example);
    }
    else
    {
        return predict(tree -> right, example);
    }
}

// DecisionTree的单独使用与测试方法：
// int main()
// {
//     Data* data = LoadData("/Users/liushihao/Desktop/搜索引擎基础/GBDT/bikeSpeedVsIq_train.txt");
//     FeaturesLabels training_set = split_features_labels(data);
//     DecisionTree tree;
//     tree.build_tree(&tree, training_set);

//     Data* test_set = LoadData("/Users/liushihao/Desktop/搜索引擎基础/GBDT/bikeSpeedVsIq_test.txt");
//     float loss = 0;
//     for (size_t i = 0; i < (*test_set).size(); ++i)
//     {   
//         float prediction = tree.predict(&tree, (*test_set)[i]);
//         loss += pow((prediction - (*test_set)[i].back()), 2); 
//         cout << (*test_set)[i].back() << '\t' << prediction << endl;
//     }
//     cout << "loss is " << loss;
//     return 0;
// }