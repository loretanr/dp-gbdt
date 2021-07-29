#include <numeric>
#include <algorithm>
#include "dp_ensemble.h"
#include "logging.h"
#include "spdlog/spdlog.h"

extern std::ofstream verification_logfile;
extern size_t cv_fold_index;
extern bool VERIFICATION_MODE;

using namespace std;


/** Constructors */

DPEnsemble::DPEnsemble(ModelParams *parameters) : params(parameters) {};
    
DPEnsemble::~DPEnsemble() {
    for (auto tree : trees) {
        tree.delete_tree(tree.root_node);
    }
};


/** Methods */

void DPEnsemble::train(DataSet *dataset)
{
    // compute initial prediction
    this->init_score = params->task->compute_init_score(dataset->y);
    LOG_DEBUG("Training initialized with score: {1}", this->init_score);

    // second split (& shuffle), not used so far
    
    if(params->second_split) {
        // TrainTestSplit split = train_test_split_random(*dataset, 0.8f, false);
        throw runtime_error("2nd-split not yet implemented.");
    }

    // Each tree gets the full pb, as they train on distinct data
    TreeParams tree_params;
    tree_params.tree_privacy_budget = params->privacy_budget;

    // distribute training instances amongst trees
    vector<DataSet> tree_samples;
    distribute_samples(&tree_samples, dataset);
    
    // train all trees
    for(int tree_index = 0; tree_index < params->nb_trees;  tree_index++) {

        LOG_DEBUG(YELLOW("Tree {1:2d}: receives pb {2:.2f} and will train on {3} instances"),
            tree_index, tree_params.tree_privacy_budget, tree_samples[tree_index].length);
        if(VERIFICATION_MODE) {
            VERIFICATION_LOG("Tree {0} CV-Ensemble {1}", tree_index, cv_fold_index);
        }

        // compute sensitivity
        tree_params.delta_g = 3 * pow(params->l2_threshold, 2);
        tree_params.delta_v = std::min((double) (params->l2_threshold / (1 + params->l2_lambda)),
                            2 * params->l2_threshold *
                            pow(1-params->learning_rate, tree_index));



        // update/init gradients of all training instances (using last tree(s))         TODO
        vector<double> gradients;
        if(tree_index == 0) {
            vector<double> init_scores(dataset->length, this->init_score);
            gradients = params->task->compute_gradients(dataset->y, init_scores);
            // store the gradients back to their corresponding samples
            int index = 0;
            for(DataSet &dset : tree_samples) {
                for (auto row : dset.X){
                    // store each gradient next to its corresponding sample
                    dset.gradients.push_back(gradients[index]);  
                    index++;
                }
            }
            gradients.resize(index); // for validation log, TODO resolve
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
            gradients = (params->task)->compute_gradients(y_samples, y_pred);

            // store the gradients back to their corresponding samples
            vector<double>::const_iterator iter = gradients.begin();
            for (size_t i=tree_index; i<tree_samples.size(); i++) {
                vector<double> curr_grads = vector<double>(iter, iter + tree_samples[i].length);
                tree_samples[i].gradients = curr_grads;
                iter += tree_samples[i].length;
            }
        }



        // intermediate output for validation
        double sum = std::accumulate(gradients.begin(), gradients.end(), 0.0);
        sum = sum < 0 && sum >= -1e-10 ? 0 : sum;  // avoid "-0.00000.. != 0.00000.."
        LOG_INFO("GRADIENTSUM {1:.8f}", sum);
        if(VERIFICATION_MODE) {
            VERIFICATION_LOG("GRADIENTSUM {0:.8f}", sum);
        }

        // gradient-based data filtering
        if(params->gradient_filtering) {
            for(DataSet &dset : tree_samples) {
                for (auto &grad : dset.gradients){
                    grad = clamp(grad, -params->l2_threshold, params->l2_threshold);
                }
            }
        }

        // build tree, add noise to leaves
        LOG_INFO("Building tree {1}...", tree_index);
        DPTree tree = DPTree(params, &tree_params, &tree_samples[tree_index], tree_index);
        tree.fit();
        trees.push_back(tree);
        
        // print the tree if we are in debug mode
        if (spdlog::default_logger_raw()->level() <= spdlog::level::debug) {
            tree.recursive_print_tree(tree.root_node);
        }

        LOG_INFO(YELLOW("Tree {1:2d} done. Instances left: {2}"), tree_index, "XX");
    }
}


// Predict values from the ensemble of gradient boosted trees
vector<double>  DPEnsemble::predict(VVD &X)
{
    // TODO inefficient, currently recomputes predictions for all trees after each tree
    vector<double> predictions(X.size(),0);
    for (auto tree : trees) {
        vector<double> pred = tree.predict(X);
        
        std::transform(pred.begin(), pred.end(), 
            predictions.begin(), predictions.begin(), std::plus<double>());
    }

    // TODO optimize those 2 transforms in 1
    double learning_rate = params->learning_rate;
    std::transform(predictions.begin(), predictions.end(),
            predictions.begin(), [learning_rate](double &c){return c*learning_rate;});

    double innit_score = this->init_score;
    std::transform(predictions.begin(), predictions.end(),
                    predictions.begin(), [innit_score](double &c){return c+innit_score;});

    return predictions;
}


// distribute training samples amongst trees
void DPEnsemble::distribute_samples(vector<DataSet> *storage_vec, DataSet *train_set)
{
    if(params->balance_partition) {
        int num_samples = train_set->length / params->nb_trees;
        int current_index = 0;
        int remainder = train_set->length % params->nb_trees;
        int remainder_index = 0;
        // same amount for every tree
        for(int i=0; i < params->nb_trees; i++) {

            VVD x_tree = {};
            vector<double> y_tree = {};

            // also distribute remainder samples one by one
            int number_of_samples = num_samples;
            if(remainder_index < remainder){
                number_of_samples++;
                remainder_index++;
            }

            // get corresponding rows from the dataset
            for(int j=0; j<number_of_samples; j++){
                x_tree.push_back((train_set->X)[current_index]);
                y_tree.push_back((train_set->y)[current_index]);
                current_index++;
            }
            DataSet d = DataSet(x_tree,y_tree);
            (*storage_vec).push_back(d);
        }
    } else {
        throw runtime_error("non-balanced split, resp. paper formula \
            partitioning not implemented yet");
    }
}