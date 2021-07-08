#include<vector>
#include<iostream>

#include "data.h"
#include "decision_tree.h"

using namespace std;

class GBDT
{
    public:
    vector<shared_ptr<DecisionTree>> forest;
    vector<float> lr_list;
    float learning_rate;
    int iterations;
    int maximum_height;
    GBDT
    (
        float learning_rate = 0.5,
        int iterations = 10,
        int maximum_height = 3
    ): learning_rate(learning_rate), iterations(iterations), maximum_height(maximum_height) {};
    void fit(FeaturesLabels&);
    vector<float> predict(FeaturesLabels&);
};

void GBDT::fit(FeaturesLabels& dataset)
{
    double sum = 0;
    vector<float> residuals;

    // 首先是将所有label的平均值作为base
    for (size_t i = 0; i < dataset.features.size(); ++i)
    {
        sum += dataset.labels[i];
    }
    float base_prediction = sum / dataset.features.size();
    auto base_tree = make_shared<DecisionTree>(1, 5, true, -1, base_prediction);

    for (size_t i = 0; i < dataset.features.size(); ++i)
    {
        residuals.push_back(dataset.labels[i] - base_prediction);
    }
    lr_list.push_back(1);
    forest.push_back(base_tree);

    //构建多棵树形成boosting，与此同时更新残差。
    for (int i = 0; i < iterations; ++i)
    {
        auto tree = make_shared<DecisionTree>(1, 5, false, -1);
        tree -> build_tree(tree, dataset);
        for (size_t j = 0; j < dataset.features.size(); ++j)
        {
            residuals[j] -= learning_rate * tree -> predict(tree, dataset.features[j]);
        }
        dataset.labels = residuals;
        forest.push_back(tree);
        lr_list.push_back(learning_rate);
    }
}

vector<float> GBDT::predict(FeaturesLabels& dataset)
{
    double loss = 0;
    vector<float> predictions;
    for (size_t i = 0; i < dataset.features.size(); ++i)
    {
        float pred = 0;
        for (size_t j = 0; j < forest.size(); ++j)
        {
            pred += (forest[j] -> predict(forest[j], dataset.features[i])) * lr_list[j];
        }
        predictions.push_back(pred);
        loss += (pred - dataset.labels[i]) * (pred - dataset.labels[i]);
        cout << dataset.labels[i] << " === " << predictions[i] << endl;
    }
    cout << "Total loss is " << loss;
    return predictions;
}

int main()
{
    Data* data = LoadData("bikeSpeedVsIq_train.txt");
    FeaturesLabels training_set = split_features_labels(data);
    GBDT gbdt_model(0.1, 50, 3); // 参数对应为lr，iterations，maximum height
    gbdt_model.fit(training_set);

    /* data所指内存已经在split_features_labels函数中被delete，但data本身仍存在 */
    data = LoadData("bikeSpeedVsIq_test.txt"); 
    FeaturesLabels test_set = split_features_labels(data);
    gbdt_model.predict(test_set);

    return 0;
}