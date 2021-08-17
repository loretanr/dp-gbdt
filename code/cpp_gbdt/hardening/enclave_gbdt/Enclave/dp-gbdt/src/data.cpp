#include <vector>
#include <numeric>
#include <queue>
#include <algorithm>
#include <cmath>
#include "data.h"
#include "utils.h"


Scaler::Scaler(double min_val, double max_val, double fmin, double fmax, bool _scaling_required) : data_min(min_val), data_max(max_val),
        feature_min(fmin), feature_max(fmax), scaling_required(_scaling_required)
{
    double data_range = data_max - data_min;
    data_range = double_equality(data_range, 0.0) ? 1 : data_range;
    this->scale = (feature_max - feature_min) / data_range;
    this->min_ = feature_min - data_min * scale;
}


DataSet::DataSet()
{
    empty = true;
}


DataSet::DataSet(VVD _X, std::vector<double> _y) : X(_X), y(_y)
{
    if(X.size() != y.size()){
        throw std::runtime_error("dataset creation failed. X,y need equal amount of rows!");
    }
    length = (int) X.size();
    num_x_cols = (int) X[0].size();
    empty = false;
}


// scale y values to be in [lower,upper]
void DataSet::scale(ModelParams &params, double lower, double upper)
{
    // only scale in dp mode
    if(params.use_dp){
        
        // return if no scaling required (y already in [-1,1])
        bool scaling_required = false;
        for(auto elem : y) {
            if (elem < lower or elem > upper) {
                scaling_required = true; break;
            }
        }
        if (not scaling_required) {
            scaler = Scaler(0,0,0,0,false);
            return;
        }

        double doublemax = std::numeric_limits<double>::max();
        double doublemin = std::numeric_limits<double>::min();
        double minimum_y = doublemax, maximum_y = doublemin;
        for(int i=0; i<length; i++) {
            minimum_y = std::min(minimum_y, y[i]);
            maximum_y = std::max(maximum_y, y[i]);
        }
        for(int i=0; i<length; i++) {
            y[i] = (y[i]- minimum_y)/(maximum_y-minimum_y) * (upper-lower) + lower;
        }
        scaler = Scaler(minimum_y, maximum_y, lower, upper, true);
    }
}


void inverse_scale(ModelParams &params, Scaler &scaler, std::vector<double> &vec)
{
    if(params.use_dp){
        // return if no scaling required
        if(not scaler.scaling_required){
            return;
        }

        for(auto &elem : vec) {
            elem -= scaler.min_;
            elem /= scaler.scale;
        }
    }
}


TrainTestSplit train_test_split_random(DataSet &dataset, double train_ratio, bool shuffle)
{
    if(shuffle) {
        dataset.shuffle_dataset();
    }

    // [ test |      train      ]
    int border = (int) ceil((1-train_ratio) * (double) dataset.y.size());

    VVD x_test(dataset.X.begin(), dataset.X.begin() + border);
    std::vector<double> y_test(dataset.y.begin(), dataset.y.begin() + border);
    VVD x_train(dataset.X.begin() + border, dataset.X.end());
    std::vector<double> y_train(dataset.y.begin() + border, dataset.y.end());

    if(train_ratio >= 1) {
        DataSet train(x_train, y_train);
        return TrainTestSplit(train, DataSet());
    } else if (train_ratio <= 0) {
        DataSet test(x_test, y_test);
        return TrainTestSplit(DataSet(), test);
    } else {
        DataSet train(x_train, y_train);
        DataSet test(x_test, y_test);
        return TrainTestSplit(train, test);
    }
}

// "reverse engineered" the python sklearn.model_selection.cross_val_score
// Returns a std::vector of the train-test-splits. Will by default shuffle 
// the dataset rows, unless we're in verification mode.
std::vector<TrainTestSplit *> create_cross_validation_inputs(DataSet *dataset, int folds)
{
    bool shuffle = true;
    if(shuffle) {
        dataset->shuffle_dataset();
    }

    int fold_size = dataset->length / folds;
    std::vector<int> fold_sizes(folds, fold_size);
    int remainder = dataset->length % folds;
    int index = 0;
    while (remainder != 0) {
        fold_sizes[index++]++;
        remainder--;
    }
    // each entry in "indices" marks a start of the test set
    // ->     [ test |        train          ]
    //                      ...
    //        [   train..   | test |  ..train ]
    //                      ...
    std::deque<int> indices(folds);
    std::partial_sum(fold_sizes.begin(), fold_sizes.end(), indices.begin());
    indices.push_front(0); 
    indices.pop_back();

    std::vector<TrainTestSplit *> splits;

    for(int i=0; i<folds; i++) {

        // don't waste memory by using local copies of the vectors.
        // work directly on what will be used.
        TrainTestSplit *split = new TrainTestSplit();
        DataSet *train = &split->train;
        DataSet *test = &split->test;

        VVD::iterator x_iterator = (dataset->X).begin() + indices[i];
        std::vector<double>::iterator y_iterator = (dataset->y).begin() + indices[i];

        // extracting the test slice is easy
        test->X = VVD(x_iterator, x_iterator + fold_sizes[i]);
        test->y = std::vector<double>(y_iterator, y_iterator + fold_sizes[i]);

        // building the train set from the remaining rows is slightly more tricky
        // (if you don't want to waste memory)
        if(i != 0){     // part before the test slice
            train->X = VVD((dataset->X).begin(), (dataset->X).begin() + indices[i]);
            train->y = std::vector<double>((dataset->y).begin(), (dataset->y).begin() + indices[i]);
        }
        if(i < folds-1){    //part after the test slice
            for(int cur_row = indices[i+1]; cur_row < dataset->length; cur_row++){
                train->X.push_back(dataset->X[cur_row]);
                train->y.push_back(dataset->y[cur_row]);
            }
        }
        // don't forget to add the meta information
        train->length = (int) train->X.size();
        train->num_x_cols = (int) train->X[0].size();
        train->empty = false;
        test->length = (int) test->X.size();
        test->num_x_cols = (int) test->X[0].size();
        test->empty = false;

        splits.push_back(split);
    }
    return splits;
}


void DataSet::shuffle_dataset()
{
    std::vector<int> indices(length);
    std::iota(std::begin(indices), std::end(indices), 0);
    sgx_vector_shuffle(indices);
    DataSet copy = *this;
    for(size_t i=0; i<indices.size(); i++){
        X[i] = copy.X[indices[i]];
        y[i] = copy.y[indices[i]];
        if (not gradients.empty()) {
            gradients[i] = copy.gradients[i];
        }
    }
}


DataSet DataSet::get_subset(std::vector<int> &indices)
{
    DataSet dataset;
    for (int i=0; i<length; i++) {
        if (std::find(indices.begin(), indices.end(), i) != indices.end()) {
            dataset.X.push_back(X[i]);
            dataset.y.push_back(y[i]);
            dataset.gradients.push_back(gradients[i]);
        }
    }
    dataset.length = (int) dataset.y.size();
    dataset.num_x_cols = (int) dataset.X[0].size();
    dataset.empty = false;
    return dataset;
}


DataSet DataSet::remove_rows(std::vector<int> &indices)
{
    DataSet dataset;
    for (int i=0; i<length; i++) {
        if (std::find(indices.begin(), indices.end(), i) == indices.end()) {
            dataset.X.push_back(X[i]);
            dataset.y.push_back(y[i]);
            dataset.gradients.push_back(gradients[i]);
        }
    }
    dataset.length = (int) dataset.y.size();
    dataset.num_x_cols = (int) X[0].size();
    dataset.empty = dataset.length == 0;
    dataset.scaler = scaler;
    return dataset;
}
