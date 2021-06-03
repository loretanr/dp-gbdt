#include "utils.h"

vector<string> split_string(const string &s, char delim) {
    vector<string> result;
    stringstream ss(s);
    string item;

    while (getline(ss, item, delim)) {
        result.push_back(item);
    }

    return result;
}

DataSet::DataSet(vector<vector<float>> X, vector<float> y) : X(X), y(y) {
    if(X.size() != y.size()){
        //throw runtime_error(string_format("X %i and y %i need equal amount of rows!", X.size(), y.size()));
        cout << X.size() << " " << y.size() << endl;
    }
    length = X.size();
}

TrainTestSplit train_test_split_random(DataSet dataset, float train_ratio, bool shuffle)
{
    if(shuffle) {
        srand(time(0));
        random_shuffle(dataset.X.begin(), dataset.X.end());
        random_shuffle(dataset.y.begin(), dataset.y.end());
    }

    int border = round(train_ratio * dataset.y.size());

    vector<vector<float>> x_train(dataset.X.begin(),dataset.X.begin() + border);
    vector<float> y_train(dataset.y.begin(),dataset.y.begin() + border);
    vector<vector<float>> x_test(dataset.X.begin() + border,dataset.X.end());
    vector<float> y_test(dataset.y.begin() + border,dataset.y.end());

    DataSet train(x_train, y_train);
    DataSet test(x_test, y_test);

    return TrainTestSplit(train, test);
}