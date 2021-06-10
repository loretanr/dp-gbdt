#include "dp_ensemble.h"


DPEnsemble::DPEnsemble(ModelParams *parameters)
{
    params = *parameters; // local copy for now
}
    

DPEnsemble::~DPEnsemble() {}; // TODO


void DPEnsemble::train(DataSet *dataset)
{
    // float prev_score = numeric_limits<float>::max();

    // init predictions
    vector<float> gradients[dataset->length];
    double sum = std::accumulate((dataset->y).begin(), (dataset->y).end(), 0.0);
    float mean = sum / dataset->length;
    std::fill(gradients->begin(), gradients->end(), mean);

    // second split (& shuffle)
    TrainTestSplit split = train_test_split_random(*dataset, 0.75f, false); // no shuffle for debug
    DataSet *train_set, *test_set;
    if(params.second_split) {
        train_set = &split.train;
        test_set = &split.test;
    } else {
        train_set = dataset;
    }

    // prepare privacy budgets
    float tree_privacy_budget = params.privacy_budget / params.nb_trees;

    // distribute training instances amongst trees
    vector<DataSet> tree_samples;
    distribute_samples(&tree_samples, train_set);
    
    // train all trees
    for(int tree_index = 0; tree_index<params.nb_trees;  tree_index++) {

        // init the dataset
        if(tree_index == 0) {
            // TODO, do I really need to work on a copy?
            // don't think so; using an array to mark deleted rows
        }

        // compute sensitivity
        params.delta_g = 3 * pow(params.l2_threshold, 2); // todo move out of loop
        params.delta_v = min((double) (params.l2_threshold / (1 + params.l2_lambda)),
                            2 * params.l2_threshold *
                            pow(1-params.learning_rate, tree_index));

        // update/init gradients of all training instances
        if(tree_index == 0) {
            double sum = std::accumulate((train_set->y).begin(), (train_set->y).end(), 0.0);
            float mean = sum / train_set->length;
            for(DataSet &dset : tree_samples) {
                for (auto row : dset.X){
                    dset.gradients.push_back(mean);
                }
            }
        } else {
            vector<float> y_pred = predict(&tree_samples[tree_index].X);
            // calculate gradient from that
            // TODO once we built the first tree.
        }

        // gradient-based data filtering
        if(params.gradient_filtering) {
            for(DataSet &dset : tree_samples) {
                for (auto &grad : dset.gradients){
                    grad = clip(grad, -params.l2_threshold, params.l2_threshold);
                }
            }
        }

        DPTree tree = DPTree(&params, &tree_samples[tree_index], tree_privacy_budget);

        cout << "training tree " << tree_index << endl;



        


    }

    trees = {};
}


vector<float>  DPEnsemble::predict(vector<vector<float>> *X)
{
    vector<float> predictions(X->size());
    for(auto tree : trees) {
        //rrrr = tree.predict(X);
        // TODO once we have the stuff to build trees.
    }


}


// distribute training instances amongst trees
void DPEnsemble::distribute_samples(vector<DataSet> *storage_vec, DataSet *train_set)
{
    if(params.balance_partition) { // perfect split
        int quotient = train_set->length / params.nb_trees;
        int remainder = train_set->length % params.nb_trees;
        int current_index = 0;
        // same amount for every tree
        for(int i=0; i<params.nb_trees; i++) {
            vector<vector<float>> x_tree = {};
            vector<float> y_tree = {};
            for(int j=0; j<quotient; j++){
                x_tree.push_back((train_set->X)[current_index]);
                y_tree.push_back((train_set->y)[current_index]);
                current_index++;
            }
            DataSet d = DataSet(x_tree,y_tree);
            (*storage_vec).push_back(d);
        }
        // evenly distribute leftover training instances
        for(int i=0; i<remainder; i++) {
            (*storage_vec)[i].add_row((train_set->X)[current_index], 
                                    (train_set->y)[current_index]);
            current_index++;
        }
    } else {
        throw runtime_error("non-balanced resp. textbook formula partition not implemented yet");
    }
}