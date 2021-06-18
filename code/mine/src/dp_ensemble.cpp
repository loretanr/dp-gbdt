#include "dp_ensemble.h"
#include "spdlog/spdlog.h"
#include "dp_tree.h"


DPEnsemble::DPEnsemble(ModelParams *parameters)
{
    params = *parameters; // local copy for now
}
    

DPEnsemble::~DPEnsemble() {
    for (auto tree : trees) {
        tree.delete_tree(tree.root_node);
    }
}; // TODO


void DPEnsemble::train(DataSet *dataset)
{
    // float prev_score = numeric_limits<float>::max();

    // init score (= mean)
    vector<float> gradients[dataset->length];
    double sum = std::accumulate((dataset->y).begin(), (dataset->y).end(), 0.0);
    float init_score = sum / dataset->length; // this is correct
    std::fill(gradients->begin(), gradients->end(), init_score);
    LOG_DEBUG("Training initialized with score: {1}", init_score);

    // second split (& shuffle), alltrees & noshuffle for debug
    TrainTestSplit split = train_test_split_random(*dataset, 1.0f, false);
    DataSet *train_set, *test_set;
    if(params.second_split) {
        train_set = &split.train;
        test_set = &split.test;
    } else {
        train_set = dataset;
    }

    TreeParams tree_params;

    // Each tree gets the full pb, as they train on distinct data
    tree_params.tree_privacy_budget = params.privacy_budget;

    // distribute training instances amongst trees
    vector<DataSet> tree_samples;
    distribute_samples(&tree_samples, train_set);
    
    // train all trees
    for(int tree_index = 0; tree_index<params.nb_trees;  tree_index++) {

        LOG_DEBUG(BOLD("Tree {1:2d}") + ": receives pb {2:.2f} and will train on {3} instances", tree_index, tree_params.tree_privacy_budget, tree_samples[tree_index].length);

        // init the dataset
        if(tree_index == 0) {
            // TODO, do I really need to work on a copy?
            // don't think so; using an array to mark deleted rows
        }

        // compute sensitivity
        tree_params.delta_g = 3 * pow(params.l2_threshold, 2); // todo move out of loop
        tree_params.delta_v = min((double) (params.l2_threshold / (1 + params.l2_lambda)),
                            2 * params.l2_threshold *
                            pow(1-params.learning_rate, tree_index));

        // update/init gradients of all training instances (using last tree(s))
        if(tree_index == 0) {
            vector<float> gradients = compute_gradient_for_loss(train_set->y, init_score);
            int index = 0;
            for(DataSet &dset : tree_samples) {
                for (auto row : dset.X){
                    // store each gradient next to its corresponding sample
                    dset.gradients.push_back(gradients[index]);  
                    index++;
                }
            }
        } else {
            vector<float> y_pred = predict(&tree_samples[tree_index].X);
            // calculate gradient from that
            // TODO once we built the first tree.
        }

        // gradient-based data filtering // (TODO untested)
        if(params.gradient_filtering) {
            for(DataSet &dset : tree_samples) {
                for (auto &grad : dset.gradients){
                    grad = clip(grad, -params.l2_threshold, params.l2_threshold);
                }
            }
        }

        // TODO, we have the right data, now build tree
        DPTree tree = DPTree(&params, &tree_params, &tree_samples[tree_index]);
        tree.fit();




        


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
    return predictions;


}

vector<float> DPEnsemble::compute_gradient_for_loss(vector<float> y, float init_score)
{
    // we want the positive gradient
    std::for_each(y.begin(), y.end(), [init_score](float& f) { f = init_score - f;});
    return y;
}


// distribute training instances amongst trees
void DPEnsemble::distribute_samples(vector<DataSet> *storage_vec, DataSet *train_set)
{
    if(params.balance_partition) { // perfect split
        int quotient = train_set->length / params.nb_trees;
        //int remainder = train_set->length % params.nb_trees;
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
        // evenly distribute leftover training instances   // TODO, disabled for debug
        // for(int i=0; i<remainder; i++) {
        //     (*storage_vec)[i].add_row((train_set->X)[current_index], 
        //                             (train_set->y)[current_index]);
        //     current_index++;
        // }
    } else {
        throw runtime_error("non-balanced resp. textbook formula partition not implemented yet");
    }
}