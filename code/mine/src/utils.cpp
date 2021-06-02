#include "utils.h"


TrainTestSplit train_test_split_random(DataSet dataset, float train_ratio)
{
    srand(time(0));
    random_shuffle(dataset.X.begin(), dataset.X.end());
    random_shuffle(dataset.y.begin(), dataset.y.end());

    int border = round(train_ratio * dataset.y.size());

    vector<vector<float>> x_train(dataset.X.begin(),dataset.X.begin() + border);
    vector<float> y_train(dataset.y.begin(),dataset.y.begin() + border);
    vector<vector<float>> x_test(dataset.X.begin() + border,dataset.X.end());
    vector<float> y_test(dataset.y.begin() + border,dataset.y.end());

    DataSet train(x_train, y_train);
    DataSet test(x_test, y_test);

    return TrainTestSplit(train, test);
}