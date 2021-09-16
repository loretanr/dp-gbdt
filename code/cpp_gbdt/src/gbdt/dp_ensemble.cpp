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
        std::call_once(flag1, [](){std::cout << "!!! DP disabled !!! (slower than dp!)" << std::endl;});
        params->use_dp = false;
        params->privacy_budget = 0;
    }

    // prepare the linspace grid
    if(params->use_grid) {
        double grid_range = std::get<1>(params->grid_borders) - std::get<0>(params->grid_borders);
        double step_size = params->grid_step_size;
        int grid_size = (int) grid_range / step_size;
        this->grid = std::vector<double>(grid_size,0);
        int counter = 0;
        std::generate(this->grid.begin(), this->grid.end(), [&counter, &step_size]() mutable { return counter++ * step_size; });
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
 
        if(VERIFICATION_MODE) {
            VERIFICATION_LOG("Tree {0} CV-Ensemble {1}", tree_index, cv_fold_index);
        }

         // update/init gradients
        update_gradients(dataset->gradients, tree_index);

        if(params->use_dp){   // build a dp-tree

            // sensitivity for internal nodes
            tree_params.delta_g = 3 * pow(params->l2_threshold, 2);

            // sensitivity for leaves
            if (params->gradient_filtering && !params->leaf_clipping) {
                // you can only "turn off" leaf clipping if GDF is enabled!
                tree_params.delta_v = params->l2_threshold / (1 + params->l2_lambda);
            } else {
                tree_params.delta_v = std::min((double) (params->l2_threshold / (1 + params->l2_lambda)),
                        2 * params->l2_threshold * pow(1-params->learning_rate, tree_index));
            }

            // determine number of rows
            int number_of_rows = 0;
            if (params->balance_partition) {
                // num_unused_rows / num_remaining_trees
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

            vector<int> tree_indices;

            // gradient-based data filtering
            if(params->gradient_filtering) {
                std::vector<int> reject_indices, remaining_indices;
                for (int i=0; i<dataset->length; i++) {
                    double curr_grad = dataset->gradients[i];
                    if (curr_grad < -params->l2_threshold or curr_grad > params->l2_threshold) {
                        reject_indices.push_back(i);
                    } else {
                        remaining_indices.push_back(i);
                    }
                }
                LOG_INFO("GDF: {1} of {2} rows fulfill gradient criterion",
                    remaining_indices.size(), dataset->length);

                if ((size_t) number_of_rows <= remaining_indices.size()) {
                    // we have enough samples that were not filtered out
                    if (!VERIFICATION_MODE) {
                        std::random_shuffle(remaining_indices.begin(), remaining_indices.end());
                    }
                    for(int i=0; i<number_of_rows; i++){
                        tree_indices.push_back(remaining_indices[i]);
                    }
                } else {
                    // we don't have enough -> take all samples that were not filtered out
                    // and fill up with randomly chosen and clipped filtered ones
                    for(auto filtered : remaining_indices){
                        tree_indices.push_back(filtered);
                    }
                    LOG_INFO("GDF: filling up with {1} rows (clipping those gradients)",
                        number_of_rows - tree_indices.size());
                    if (!VERIFICATION_MODE) {
                        std::random_shuffle(reject_indices.begin(), reject_indices.end());
                    }
                    int reject_index = 0;
                    for(int i=tree_indices.size(); i<number_of_rows; i++){
                        int curr_index = reject_indices[reject_index++];
                        dataset->gradients[curr_index] = clamp(dataset->gradients[curr_index],
                            -params->l2_threshold, params->l2_threshold);
                        tree_indices.push_back(curr_index);
                    }
                }
            } else {
                // no GDF, just randomly select <number_of_rows> rows.
                // Note, this causes the leaves to be clipped after building the tree.
                tree_indices = vector<int>(dataset->length);
                std::iota(std::begin(tree_indices), std::end(tree_indices), 0);
                if (!VERIFICATION_MODE) {
                    std::random_shuffle(tree_indices.begin(), tree_indices.end());
                }
                tree_indices = std::vector<int>(tree_indices.begin(), tree_indices.begin() + number_of_rows);
            }

            DataSet tree_dataset = dataset->get_subset(tree_indices);
            
            LOG_DEBUG(YELLOW("Tree {1:2d}: receives pb {2:.2f} and will train on {3} instances"),
                    tree_index, tree_params.tree_privacy_budget, tree_dataset.length);

            // build tree
            LOG_INFO("Building dp-tree-{1} using {2} samples...", tree_index, tree_dataset.length);
            DPTree tree = DPTree(params, &tree_params, &tree_dataset, tree_index, this->grid);
            // DPTree tree = DPTree(params, &tree_params, dataset, tree_index);
            tree.fit();
            trees.push_back(tree);

            // remove rows
            *dataset = dataset->remove_rows(tree_indices);

        } else {  // build a non-dp tree
            
            LOG_DEBUG(YELLOW("Tree {1:2d}: receives pb {2:.2f} and will train on {3} instances"),
                    tree_index, tree_params.tree_privacy_budget, dataset->length);

            // build tree
            LOG_INFO("Building non-dp-tree {1} using {2} samples...", tree_index, dataset->length);
            DPTree tree = DPTree(params, &tree_params, dataset, tree_index, this->grid);
            tree.fit();
            trees.push_back(tree);
        }

        // print the tree if we are in debug mode
        if (spdlog::default_logger_raw()->level() <= spdlog::level::debug) {
            trees.back().recursive_print_tree(trees.back().root_node);
        }
        LOG_INFO(YELLOW("Tree {1:2d} done. Instances left: {2}"), tree_index, dataset->length);
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
    if(VERIFICATION_MODE) {
        double sum = std::accumulate(gradients.begin(), gradients.end(), 0.0);
        sum = sum < 0 && sum >= -1e-10 ? 0 : sum;  // avoid "-0.00000.. != 0.00000.."
        VERIFICATION_LOG("GRADIENTSUM {0:.8f}", sum);
    }
}
