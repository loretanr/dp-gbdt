#include <numeric>
#include <algorithm>
#include <iostream>
#include "dp_ensemble.h"
#include "constant_time.h"
#include "logging.h"
#include "spdlog/spdlog.h"
#include "utils.h"

extern std::ofstream verification_logfile;
extern size_t cv_fold_index;
extern bool VERIFICATION_MODE;

using namespace std;


/** Constructors */

DPEnsemble::DPEnsemble(ModelParams *parameters) : params(parameters)
{
    if (parameters->privacy_budget == 0){
        throw std::runtime_error("hardened gbdt cannot be run with pb=0");
    }
}
    
DPEnsemble::~DPEnsemble() {
    for (auto tree : trees) {
        tree.delete_tree();
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
    
    // train all trees
    for(int tree_index = 0; tree_index < params->nb_trees;  tree_index++) {
 
        if(VERIFICATION_MODE) {
            VERIFICATION_LOG("Tree {0} CV-Ensemble {1}", tree_index, cv_fold_index);
        }

        // update/init gradients
        update_gradients(dataset->gradients, tree_index);

        /* build a dp-tree */

        // sensitivity for internal nodes
        tree_params.delta_g = 3 * pow(params->l2_threshold, 2);

        // sensitivity for leaves
        if (is_true(params->gradient_filtering) && !is_true(params->leaf_clipping)) {
            // you can only "turn off" leaf clipping if GDF is enabled!
            tree_params.delta_v = params->l2_threshold / (1 + params->l2_lambda);
        } else {
            tree_params.delta_v = std::min((double) (params->l2_threshold / (1 + params->l2_lambda)),
                    2 * params->l2_threshold * pow(1-params->learning_rate, tree_index));
        }

        // determine number of rows
        int number_of_rows = 0;
        if (is_true(params->balance_partition)) {
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

        vector<int> tree_indices(dataset->length);

        // gradient-based data filtering
        if(is_true(params->gradient_filtering)) {
            std::vector<int> reject_indices(dataset->length,0), remaining_indices(dataset->length,0);
            for (int i=0; i<dataset->length; i++) {
                double curr_grad = dataset->gradients[i];
                bool reject = curr_grad < -params->l2_threshold or curr_grad > params->l2_threshold; // TODO
                if (reject) {
                    reject_indices[i] = 1;          // TODO
                } else {
                    remaining_indices[i] = 1;
                }
            }
            int remaining_count = std::count(remaining_indices.begin(), remaining_indices.end(), 1);
            int reject_count = std::count(reject_indices.begin(), reject_indices.end(), 1);
            LOG_INFO("GDF: {1} of {2} rows fulfill gradient criterion",
                remaining_count, dataset->length);

            if (number_of_rows <= remaining_count) {                            // TODO need to hide this
                // we have enough samples that were not filtered out

                // generate random index permutation
                std::vector<int> shuffler(remaining_indices.size());
                std::iota(std::begin(shuffler), std::end(shuffler), 1); // indices start at 1
                std::random_shuffle(shuffler.begin(), shuffler.end());
                // cancel out rejected ones
                for(auto &elem : shuffler){
                    elem *= remaining_indices[elem-1];
                }
                std::transform(shuffler.begin(), shuffler.end(), shuffler.begin(), [](int &c){return c-1;}); // make it 0-indexed

                // construct tree_indices
                int taken_rows = 0;
                for(size_t i=0; i<shuffler.size(); i++){
                    bool use_it = constant_time::logical_and(shuffler[i] != -1, taken_rows < number_of_rows);
                    taken_rows += (int) use_it;
                    // touch the entire tree_samples vector, to hide which one is added
                    for(int j=0; j < (int) tree_indices.size(); j++){
                        bool right_element = constant_time::logical_and(j == shuffler[i], use_it);
                        tree_indices[j] = constant_time::select(right_element, 1, tree_indices[j]);
                    }
                }

            } else {
                // we don't have enough -> take all samples that were not filtered out
                // and fill up with randomly chosen and clipped filtered ones
                for(size_t i=0; i<remaining_indices.size(); i++) {
                    for(size_t j=0; j<tree_indices.size(); j++){
                        tree_indices[j] = constant_time::select(j == i, remaining_indices[i], tree_indices[j]);
                    }
                }

                int num_additional_samples_required = number_of_rows - remaining_count;

                // generate random index permutation
                std::vector<int> shuffler(reject_indices.size());
                std::iota(std::begin(shuffler), std::end(shuffler), 1); // indices start at 1
                std::random_shuffle(shuffler.begin(), shuffler.end());
                // cancel out non-rejected ones
                for(auto &elem : shuffler){
                    elem *= reject_indices[elem-1];
                }
                std::transform(shuffler.begin(), shuffler.end(), shuffler.begin(), [](int &c){return c-1;}); // make it 0-indexed
                // fill up tree_indices, loop over all elements to hide how many we fill up with
                int fill_counter = 0;
                for(size_t i=0; i<shuffler.size(); i++) {
                    bool use_it = constant_time::logical_and(shuffler[i] != -1, fill_counter < num_additional_samples_required);
                    fill_counter += (int) use_it;
                    // touch the entire tree_samples vector, to hide which one is added
                    for(int j=0; j < (int) tree_indices.size(); j++){
                        bool right_element = constant_time::logical_and(j == shuffler[i], use_it);
                        tree_indices[j] = constant_time::select(right_element, 1, tree_indices[j]);
                    }
                }
            }
        } else { // no need to harden this, right?

            // no GDF, just randomly select <number_of_rows> rows.
            // Note, this causes the leaves to be clipped after building the tree.
            vector<int> some_indices(dataset->length);
            std::iota(std::begin(some_indices), std::end(some_indices), 0);
            if (!VERIFICATION_MODE) {
                std::random_shuffle(some_indices.begin(), some_indices.end());
            }
            for(int i=0; i< number_of_rows; i++){       
                tree_indices[some_indices[i]] = 1;
            }            
        }

        DataSet tree_dataset = dataset->get_subset(tree_indices);   // TODO
        
        LOG_DEBUG(YELLOW("Tree {1:2d}: receives pb {2:.2f} and will train on {3} instances"),
                tree_index, tree_params.tree_privacy_budget, tree_dataset.length);

        // build tree
        LOG_INFO("Building dp-tree-{1} using {2} samples...", tree_index, tree_dataset.length);
        DPTree tree = DPTree(params, &tree_params, &tree_dataset, tree_index);
        tree.fit();
        trees.push_back(tree);

        // remove rows
        *dataset = dataset->remove_rows(tree_indices);  // TODO

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
