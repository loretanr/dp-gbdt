#include "utils.h"

vector<string> split_string(const string &s, char delim)
{
    vector<string> result;
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        result.push_back(item);
    }
    return result;
}


float clip(float n, float lower, float upper)
{
  return std::max(lower, std::min(n, upper));
}


template <typename Iter>
typename std::iterator_traits<Iter>::value_type log_sum_exp(Iter begin, Iter end)
{
  using VT = std::iterator_traits<Iter>::value_type{};
  if (begin==end) return VT{};
  using std::exp;
  using std::log;
  auto max_elem = *std::max_element(begin, end);
  auto sum = std::accumulate(begin, end, VT{}, 
     [max_elem](VT a, VT b) { return a + exp(b - max_elem); });
  return max_elem + log(sum);
}


DataSet::DataSet()
{
    empty = true;
}

Scaler::Scaler() {}


DataSet::DataSet(vector<vector<float>> X, vector<float> y) : X(X), y(y)
{
    if(X.size() != y.size()){
        stringstream message;
        message << "X,y need equal amount of rows! (" << X.size() << ',' << y.size() << ')';
        throw runtime_error(message.str());
    }
    length = X.size();
    num_x_cols = X[0].size();
    empty = false;
}


void DataSet::add_row(vector<float> xrow, float yval)
{
    this->X.push_back(xrow);
    this->y.push_back(yval);
    length++;
}


// scale the features such that they lie in [lower,upper]
// !!! Seems like only y needs to be scaled !!!
void DataSet::scale(float lower, float upper)
{
    float floatmax = numeric_limits<float>::max();
    float floatmin = numeric_limits<float>::min();
    // vector<float> minima_x(num_x_cols,floatmax), maxima_x(num_x_cols,floatmin);
    float minimum_y = floatmax, maximum_y = floatmin;
    for(int i=0; i<length; i++) {
        // for(int j=0; j<num_x_cols; j++) {
        //     minima_x[j] = std::min(minima_x[j], X[i][j]);
        //     maxima_x[j] = std::max(maxima_x[j], X[i][j]);
        // }
        minimum_y = std::min(minimum_y, this->y[i]);
        maximum_y = std::max(maximum_y, this->y[i]);
    }
    for(int i=0; i<length; i++) {
        // only scale numerical features of X
        // for(auto j : params.num_idx) {
        //     X[i][j] = (X[i][j]-minima_x[j])/(maxima_x[j]-minima_x[j]) * (upper-lower) + lower; 
        // }
        // scale y as well
        this->y[i] = (this->y[i]- minimum_y)/(maximum_y-minimum_y) * (upper-lower) + lower;
    }
    this->scaler = Scaler(minimum_y, maximum_y, lower, upper);
}

TrainTestSplit train_test_split_random(DataSet dataset, float train_ratio, bool shuffle)
{
    if(shuffle) {
        srand(time(0));
        random_shuffle(dataset.X.begin(), dataset.X.end());
        random_shuffle(dataset.y.begin(), dataset.y.end());
    }

    // [ test |      train      ]
    int border = ceil((1-train_ratio) * dataset.y.size());

    vector<vector<float>> x_test(dataset.X.begin(), dataset.X.begin() + border);
    vector<float> y_test(dataset.y.begin(), dataset.y.begin() + border);
    vector<vector<float>> x_train(dataset.X.begin() + border, dataset.X.end());
    vector<float> y_train(dataset.y.begin() + border, dataset.y.end());

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