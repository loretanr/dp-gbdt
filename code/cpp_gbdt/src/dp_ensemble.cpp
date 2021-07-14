#include "dp_ensemble.h"



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

    // init score (= mean)
    vector<double> gradients[dataset->length];
    double sum = std::accumulate((dataset->y).begin(), (dataset->y).end(), 0.0);
    double init_score = sum / dataset->length; // this is correct
    this->init_score = init_score;
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

        LOG_DEBUG(BOLD("Tree {1:2d}: receives pb {2:.2f} and will train on {3} instances"), tree_index, tree_params.tree_privacy_budget, tree_samples[tree_index].length);
        if(VERIFICATION_MODE) {
            VERIFICATION_LOG("Tree {0} CV-Ensemble {1}", tree_index, cv_fold_index);
        }

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
        vector<double> gradients;
        if(tree_index == 0) {
            vector<double> init_scores(train_set->length, init_score);
            gradients = compute_gradient_for_loss(train_set->y, init_scores);
            int index = 0;
            for(DataSet &dset : tree_samples) {
                for (auto row : dset.X){
                    // store each gradient next to its corresponding sample
                    dset.gradients.push_back(gradients[index]);  
                    index++;
                }
            }
            gradients.resize(index); // for validation log
        } else {
            // only have to update gradients of unused samples
            VVD pred_samples;
            vector<double> y_samples;
            for (size_t i=tree_index; i<tree_samples.size(); i++) {
                pred_samples.insert(pred_samples.end(), tree_samples[i].X.begin(), tree_samples[i].X.end());
                y_samples.insert(y_samples.end(), tree_samples[i].y.begin(), tree_samples[i].y.end());
            }
            vector<double> y_pred = predict(pred_samples);

            // update gradients
            gradients = compute_gradient_for_loss(y_samples, y_pred);

            // store them
            vector<double>::const_iterator iter = gradients.begin();
            for (size_t i=tree_index; i<tree_samples.size(); i++) {
                vector<double> curr_grads = vector<double>(iter, iter + tree_samples[i].length);
                tree_samples[i].gradients = curr_grads;
                iter += tree_samples[i].length;
            }
        }
        // intermediate output for validation
        double sum = std::accumulate(gradients.begin(), gradients.end(), 0.0);
        LOG_INFO("GRADIENTSUM {1:.8f}", sum);
        if(VERIFICATION_MODE) {
            VERIFICATION_LOG("GRADIENTSUM {0:.10f}", sum);
        }

        // gradient-based data filtering
        if(params.gradient_filtering) {
            for(DataSet &dset : tree_samples) {
                for (auto &grad : dset.gradients){
                    grad = clip(grad, -params.l2_threshold, params.l2_threshold);
                }
            }
        }

        // TODO, we have the right data, now build tree
        DPTree tree = DPTree(&params, &tree_params, &tree_samples[tree_index], tree_index);
        
        // build tree, add noise to leaves
        LOG_INFO("Building tree {1}...", tree_index);

        tree.fit();

        trees.push_back(tree);
        // tree.recursive_print_tree(tree.root_node);


        LOG_INFO(BOLD("Tree {1:2d} done. Instances left: {2}"), tree_index, "XX");

    }

}

// Predict values from the ensemble of gradient boosted trees
vector<double>  DPEnsemble::predict(VVD &X)
{
    vector<double> predictions(X.size(),0);
    for (auto tree : trees) {
        vector<double> pred = tree.predict(X);

        // if(X.size() == 836) {
        //     cout << pred[0] << " ";
        // }
        
        std::transform(pred.begin(), pred.end(), 
            predictions.begin(), predictions.begin(), std::plus<double>());
    }

    // todo optimize those in 1
    double learning_rate = params.learning_rate;
    std::transform(predictions.begin(), predictions.end(),
            predictions.begin(), [learning_rate](double &c){return c*learning_rate;});

    double innit_score = this->init_score;
    std::transform(predictions.begin(), predictions.end(),
                    predictions.begin(), [innit_score](double &c){return c+innit_score;});

    return predictions;


}

vector<double> DPEnsemble::compute_gradient_for_loss(vector<double> y, vector<double> &scores)
{
    // we want the positive gradient
    //std::for_each(y.begin(), y.end(), [init_score](double& f) { f = init_score - f;});
    for (size_t i=0; i<y.size(); i++) {
        y[i] = scores[i] - y[i];
    }
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
            VVD x_tree = {};
            vector<double> y_tree = {};
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