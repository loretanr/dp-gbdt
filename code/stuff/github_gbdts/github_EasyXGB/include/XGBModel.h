#ifndef XGBMODEL_H
#define XGBMODEL_H

#include <vector>
#include <iostream>
#include <memory>
#include "Tree.h"

using namespace std;

class XGBModel {
public:
    XGBModel(Dataset* dataset, ModelParam params);
    void train();
    void evaluate(Dataset* dataset);
    float inference(vector<float>& instance);

private:
    void train_fold(int fold_index, float& train_avg_rmse, float& train_avg_acc,
                    float& valid_avg_rmse, float& valid_avg_acc);
    vector<shared_ptr<Tree>> trees;
    Dataset* dataset;
    ModelParam params;
};

#endif // XGBMODEL_H
