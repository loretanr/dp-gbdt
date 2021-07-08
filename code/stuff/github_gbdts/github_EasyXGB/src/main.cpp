#include <iostream>
#include <string>
#include <unordered_map>
#include <thread>

#include "XGBModel.h"

using namespace std;

int main()
{
    Dataset dataset("data/classification/data_banknote.csv",
                    vector<int>{1,1,1,1}, "binary");
    ModelParam params;
    params.num_trees = 50;
    params.max_depth = 20;
    params.num_folds = 10;
    params.min_examples_leaf = 50;
    params.learning_rate = 0.5;
    params.gamma = 0;
    params.col_sample = 1;
    params.objective = "binary"; // or regression
    params.print_tree = false;
    params.prune_tree = true;
    params.multi_threads = true;

    XGBModel model(&dataset, params);
    model.train();

    return 0;
}
