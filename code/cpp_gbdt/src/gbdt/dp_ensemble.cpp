#include <numeric>
#include <algorithm>
#include <mutex>
#include <iostream>
#include "dp_ensemble.h"
#include "logging.h"
#include "spdlog/spdlog.h"

extern std::ofstream verification_logfile;
extern size_t cv_fold_index;
extern bool VERIFICATION_MODE;
extern std::once_flag flag1;

using namespace std;


/** Constructors */

DPEnsemble::DPEnsemble(ModelParams *parameters) : params(parameters)
{
    // only output this once, in case we're running with multiple threads
    if (parameters->privacy_budget == 0 or !parameters->use_dp){
        std::call_once(flag1, [](){std::cout << "!!! DP disabled !!!" << std::endl;});
        params->use_dp = false;
        params->privacy_budget = 0;
    }
}
    
DPEnsemble::~DPEnsemble() {
    for (auto tree : trees) {
        tree.delete_tree(tree.root_node);
    }
}


/** Methods */

void DPEnsemble::train(DataSet *dataset)
{   
    this->dataset = dataset;
    int original_length = dataset->length;

    // compute initial prediction
    this->init_score = params->task->compute_init_score(dataset->y);
    LOG_DEBUG("Training initialized with score: {1}", init_score);

    // each tree gets the full pb, as they train on distinct data
    TreeParams tree_params;
    tree_params.tree_privacy_budget = params->privacy_budget;
    tree_params.delta_g = 0;
    tree_params.delta_v = 0;
    
    // train all trees
    for(int tree_index = 0; tree_index < params->nb_trees;  tree_index++) {

            
        // if(VERIFICATION_MODE) {
        //     VERIFICATION_LOG("Tree {0} CV-Ensemble {1}", tree_index, cv_fold_index);
        // }

         // update/init gradients
        update_gradients(dataset->gradients, tree_index);

        if(params->use_dp){   // build a dp-tree
            
            // compute sensitivity
            tree_params.delta_g = 3 * pow(params->l2_threshold, 2);
            tree_params.delta_v = std::min((double) (params->l2_threshold / (1 + params->l2_lambda)),
                                2 * params->l2_threshold *
                                pow(1-params->learning_rate, tree_index));

            // determine number of rows
            int number_of_rows = 0;
            if (params->balance_partition) {
                number_of_rows = dataset->length / (params->nb_trees - tree_index);
            } else {
                // line 8 of Algorithm 2 from the paper
                number_of_rows = (original_length * params->learning_rate *
                        std::pow(1 - params->learning_rate, tree_index)) / 
                        (1 - std::pow(1 - params->learning_rate, params->nb_trees));
                if (number_of_rows == 0) {
                    throw std::runtime_error("Warning: tree is not getting any samples");
                }
            }

            // randomly select the rows
            vector<int> indices(dataset->length);
            std::iota(std::begin(indices), std::end(indices), 0);
            std::random_shuffle(indices.begin(), indices.end());
            indices = std::vector<int>(indices.begin(), indices.begin() + number_of_rows);
            DataSet tree_dataset = dataset->get_subset(indices);
            
            // gradient-based data filtering  // TODO, put into subset?
            if(params->gradient_filtering) {
                int count = 0;
                std::vector<int> reject_indices;
                for (int i=0; i<tree_dataset.length; i++) {
                    int curr_grad = tree_dataset.gradients[i];
                    if (curr_grad < -params->l2_threshold or curr_grad > params->l2_threshold) {
                        reject_indices.push_back(i);
                        count++;
                    }
                }
                std::cout << fmt::format("GDF: rejecting ({}/{})", count, tree_dataset.length) << std::endl;
                tree_dataset = tree_dataset.remove_rows(reject_indices);
                for(auto index : reject_indices) {
                    indices.erase(indices.begin() + index);
                }
            }

            LOG_DEBUG(YELLOW("Tree {1:2d}: receives pb {2:.2f} and will train on {3} instances"),
                    tree_index, tree_params.tree_privacy_budget, tree_dataset.length);

            // build tree
            LOG_INFO("Building tree {1}...", tree_index);
            DPTree tree = DPTree(params, &tree_params, &tree_dataset, tree_index);
            tree.fit();
            trees.push_back(tree);

            // remove rows
            *dataset = dataset->remove_rows(indices);

            if (spdlog::default_logger_raw()->level() <= spdlog::level::debug) {
                trees.back().recursive_print_tree(trees.back().root_node);
            }
            LOG_INFO(YELLOW("Tree {1:2d} done. Instances left: {2}"), tree_index, dataset->length);

        } else {  // build a non-dp tree
            
            LOG_DEBUG(YELLOW("Tree {1:2d}: receives pb {2:.2f} and will train on {3} instances"),
                    tree_index, tree_params.tree_privacy_budget, dataset->length);

            // build tree
            LOG_INFO("Building tree {1}...", tree_index);
            DPTree tree = DPTree(params, &tree_params, dataset, tree_index);
            tree.fit();
            trees.push_back(tree);
            // print the tree if we are in debug mode

            if (spdlog::default_logger_raw()->level() <= spdlog::level::debug) {
                trees.back().recursive_print_tree(trees.back().root_node);
            }
            LOG_INFO(YELLOW("Tree {1:2d} done. Instances left: {2}"), tree_index, dataset->length);
        }
    }
}


// Predict values from the ensemble of gradient boosted trees
vector<double>  DPEnsemble::predict(VVD &X)
{
    vector<double> predictions(X.size(),0);
    for (auto tree : trees) {
        vector<double> pred = tree.predict(X);
        
        std::transform(pred.begin(), pred.end(), 
            predictions.begin(), predictions.begin(), std::plus<double>());
    }

    double innit_score = this->init_score;
    double learning_rate = params->learning_rate;
    std::transform(predictions.begin(), predictions.end(), predictions.begin(), 
            [learning_rate, innit_score](double &c){return c*learning_rate + innit_score;});

    return predictions;
}


// distribute training samples in train_set amongst trees
// by splitting into even chunks and storing them to storage_vec
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


void DPEnsemble::update_gradients(vector<double> &gradients, int tree_index)
{
    if(tree_index == 0) {
        // init gradients
        vector<double> init_scores(dataset->length, init_score);
        gradients = params->task->compute_gradients(dataset->y, init_scores);
    } else { 
        // update gradients
        vector<double> y_pred = predict(dataset->X);
        gradients = (params->task)->compute_gradients(dataset->y, y_pred);
    }
}
