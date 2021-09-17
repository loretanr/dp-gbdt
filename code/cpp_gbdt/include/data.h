#ifndef DATA_H
#define DATA_H

#include <vector>
#include <set>
#include "utils.h"

// if the target needs to be scaled (into [-1,1]) before training, we store
// everything in this struct, that is required to invert the scaling after training 
struct Scaler {
    double data_min, data_max;
    double feature_min, feature_max;
    double scale, min_;
    bool scaling_required;
    Scaler() {};
    Scaler(double min_val, double max_val, double fmin, double fmax, bool scaling_required);
};

// basic wrapper around our data:
//  - matrix X
//  - target y
//  - vector for the samples' gradients (which get constantly updated)
//  - some useful attributes
struct DataSet {
    // constructors
    DataSet();
    DataSet(VVD X, std::vector<double> y);

    // fields
    VVD X;
    std::vector<double> y;
    std::vector<double> gradients;
    int length, num_x_cols;
    bool empty;
    Scaler scaler;
    std::string name;

    // methods
    void add_row(std::vector<double> xrow, double yval);
    void scale_y(ModelParams &params, double lower, double upper);
    void scale_X_columns(ModelParams &params);
    void shuffle_dataset();
    DataSet get_subset(std::vector<int> &indices);
    DataSet remove_rows(std::vector<int> &indices);
};

// wrapper around 2 DataSets that belong together
struct TrainTestSplit {
    DataSet train;
    DataSet test;
    TrainTestSplit(DataSet train, DataSet test) : train(train), test(test) {};
    TrainTestSplit() {};
};


// method declarations
void inverse_scale_y(ModelParams &params, Scaler &scaler, std::vector<double> &vec);
TrainTestSplit train_test_split_random(DataSet &dataset, double train_ratio = 0.70, bool shuffle = false);
std::vector<TrainTestSplit *> create_cross_validation_inputs(DataSet *dataset, int folds);
// std::tuple<double,double> dp_confidence_interval(std::vector<double> samples, double percentile, double budget);


#endif /* DATA_H */
