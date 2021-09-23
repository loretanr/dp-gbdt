#include <vector>
#include <numeric>
#include <queue>
#include <random>
#include <algorithm>
#include <sstream>
#include <cmath>
#include "data.h"


extern bool VERIFICATION_MODE;


Scaler::Scaler(double min_val, double max_val, double fmin, double fmax, bool scaling_required) : data_min(min_val), data_max(max_val),
        feature_min(fmin), feature_max(fmax), scaling_required(scaling_required)
{
    double data_range = data_max - data_min;
    data_range = data_range == 0 ? 1 : data_range;
    this->scale = (feature_max - feature_min) / data_range;
    this->min_ = feature_min - data_min * scale;
}


DataSet::DataSet()
{
    empty = true;
}


DataSet::DataSet(VVD X, std::vector<double> y) : X(X), y(y)
{
    if(X.size() != y.size()){
        std::stringstream message;
        message << "X,y need equal amount of rows! (" << X.size() << ',' << y.size() << ')';
        throw std::runtime_error(message.str());
    }
    length = X.size();
    num_x_cols = X[0].size();
    empty = false;
}


// scale y values to be in [lower,upper]
void DataSet::scale_y(ModelParams &params, double lower, double upper)
{   
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


void inverse_scale_y(ModelParams &params, Scaler &scaler, std::vector<double> &vec)
{
    // return if no scaling required
    if(not scaler.scaling_required){
        return;
    }

    for(auto &elem : vec) {
        elem -= scaler.min_;
        elem /= scaler.scale;
    }
}


// algorithm (EXPQ) from this paper:
//      https://arxiv.org/pdf/2001.02285.pdf
// corresponding code:
//  https://github.com/wxindu/dp-conf-int/blob/master/algorithms/alg5_EXPQ.R
std::tuple<double,double> dp_confidence_interval(std::vector<double> &samples, double percentile, double budget)
{
    // e.g.  95% -> {0.025, 0.975}
    std::vector<double> quantiles = {(1.0-percentile/100.)/2., percentile/100. + (1.0-percentile/100.)/2.};
    std::vector<double> results;

    // set up inputs
    std::sort(samples.begin(), samples.end());
    double *db = samples.data();
    int n = samples.size();
    double e = budget / 2;  // half budget since we're doing it twice

    // run the dp quantile calculation twice (to get the lower & upper bound)
    for(auto quantile : quantiles) {

        double q = quantile;
        int m = std::floor((n-1)*q + 1.5);
        std::vector<double> probs(n+1);
        std::iota(probs.begin(), probs.end(), 1.0);   // [1,2,...,n+1]
        double r = ((double)std::rand()/(double)RAND_MAX);
        if(VERIFICATION_MODE) {
            r = 0.5;
        }
        int priv_qi = 0;
        
        // exponential mechanism
        // https://github.com/wxindu/dp-conf-int/blob/master/algorithms/exp_mech.c
        for(int i = 0; i < m; i++) {
            double utility = (i + 1) - m;
            probs[i] = std::max(0.0, (db[i + 1] - db[i])*std::exp(e * utility / 2.));
        }
        for(int i = m; i <= n; i++) {
            double utility = m - i;
            probs[i] = std::max(0.0, (db[i + 1] - db[i])*std::exp(e * utility / 2.));
        }
        double sum = 0;
        for(int i = 0; i <= n; i++) sum += probs[i];
        r *= sum;
        for(int i = 0; i <= n; i++) {
            r -= probs[i];
            if(r < 0) {
                priv_qi = i;
                break;
            }
        }
        std::uniform_real_distribution<double> unif(db[priv_qi],db[priv_qi+1]);
        std::default_random_engine re;
        double a_random_double = unif(re);
        if(VERIFICATION_MODE){
            a_random_double = db[priv_qi];
        }
        results.push_back(a_random_double);
    }
    return std::make_tuple(results[0],results[1]);
}


// scale numerical features of X to fit our grid borders
void DataSet::scale_X_columns(ModelParams &params)
{
    // in order to legally scale X into our grid, we first need to (dp-)compute 
    // the [e.g. 95%] percentile borders, and clip all outliers.
    for(int col=0; col<num_x_cols; col++){

        // ignore categorical columns
        if(std::find(params.num_idx.begin(), params.num_idx.end(), col) == params.num_idx.end()){
            continue;
        }

        // get the column
        std::vector<double> column;
        for(int row=0; row<length; row++) {
            column.push_back(X[row][col]);
        }

        // compute percentile borders on our data
        std::tuple<double,double> borders = dp_confidence_interval(column, params.scale_X_percentile, params.scale_X_privacy_budget);
        double lower = std::get<0>(borders);
        double upper = std::get<1>(borders);

        // clip outliers
        for(int row=0; row<length; row++) {
            X[row][col] = clamp(X[row][col], lower, upper);
        }

        // now we should be able to safely scale the feature into our grid
        double min_val = std::numeric_limits<double>::max();
        double max_val = std::numeric_limits<double>::min();
        for(int row=0; row<length; row++) {
            min_val = std::min(min_val, X[row][col]);
            max_val = std::max(max_val, X[row][col]);
        }
        lower = std::get<0>(params.grid_borders);
        upper = std::get<1>(params.grid_borders);
        for(int row=0; row<length; row++) {
            X[row][col] = (X[row][col]-min_val)/(max_val-min_val)*(upper-lower)+lower;
        }
    }
}


TrainTestSplit train_test_split_random(DataSet &dataset, double train_ratio, bool shuffle)
{
    if(shuffle) {
        dataset.shuffle_dataset();
    }

    // [ test |      train      ]
    int border = ceil((1-train_ratio) * dataset.y.size());

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
    bool shuffle = !VERIFICATION_MODE;
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
        train->length = train->X.size();
        train->num_x_cols = train->X[0].size();
        train->empty = false;
        test->length = test->X.size();
        test->num_x_cols = test->X[0].size();
        test->empty = false;

        splits.push_back(split);
    }
    return splits;
}


void DataSet::shuffle_dataset()
{
    std::vector<int> indices(length);
    std::iota(std::begin(indices), std::end(indices), 0);
    std::random_shuffle(indices.begin(), indices.end());
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
        if (indices[i]) {
            dataset.X.push_back(X[i]);
            dataset.y.push_back(y[i]);
            dataset.gradients.push_back(gradients[i]);
        }
    }
    dataset.length = dataset.y.size();
    dataset.num_x_cols = dataset.X[0].size();
    dataset.empty = false;
    return dataset;
}


DataSet DataSet::remove_rows(std::vector<int> &indices)
{
    DataSet dataset;
    for (int i=0; i<length; i++) {
        if (!indices[i]) {
            dataset.X.push_back(X[i]);
            dataset.y.push_back(y[i]);
            dataset.gradients.push_back(gradients[i]);
        }
    }
    dataset.length = dataset.y.size();
    dataset.num_x_cols = X[0].size();
    dataset.empty = dataset.length == 0;
    dataset.scaler = scaler;
    return dataset;
}
