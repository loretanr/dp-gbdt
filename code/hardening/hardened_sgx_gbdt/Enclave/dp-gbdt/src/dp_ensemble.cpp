#include <numeric>
#include <cmath>
#include <algorithm>
#include <mutex>
#include "dp_ensemble.h"
#include "constant_time.h"
#include "utils.h"
#include "Enclave.h"

using namespace std;


/** Constructors */

DPEnsemble::DPEnsemble(ModelParams *parameters) : params(parameters)
{
    if (double_equality(parameters->privacy_budget, 0.0) or !parameters->use_dp){
        params->use_dp = false;
        params->privacy_budget = 0;
        sgx_printf("!!! DP disabled !!! (slower than dp!)\n");
    }
}
    
DPEnsemble::~DPEnsemble() {
    for (auto tree : trees) {
        tree.delete_tree(tree.root_node);
    }
}


/** Methods */

void DPEnsemble::train(DataSet *_dataset)
{   
    this->dataset = _dataset;
    int original_length = dataset->length;

    // compute initial prediction
    this->init_score = params->task->compute_init_score(dataset->y);

    // each tree gets the full pb, as they train on distinct data
    TreeParams tree_params;
    tree_params.tree_privacy_budget = params->privacy_budget;
    tree_params.delta_g = 0;
    tree_params.delta_v = 0;
    
    // train all trees
    for(int tree_index = 0; tree_index < params->nb_trees;  tree_index++) {

         // update/init gradients
        update_gradients(dataset->gradients, tree_index);

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
            number_of_rows = dataset->length / (params->nb_trees - tree_index);
        } else {
            // line 8 of Algorithm 2 from the paper
            number_of_rows = (int) ((original_length * params->learning_rate *
                    std::pow(1 - params->learning_rate, tree_index)) / 
                    (1 - std::pow(1 - params->learning_rate, params->nb_trees)));
            if (number_of_rows == 0) {
                throw std::runtime_error("Warning: tree is not getting any samples");
            }
        }

        // this vector indicates which sample rows will be used for the next tree
        vector<int> tree_indices(dataset->length);

        // gradient-based data filtering
        if(params->gradient_filtering) {

            // divide samples into rejected/remaining gradients
            std::vector<int> reject_indices(dataset->length,0), remaining_indices(dataset->length,0);
            for (int i=0; i<dataset->length; i++) {
                double curr_grad = dataset->gradients[i];
                bool reject = constant_time::logical_or(curr_grad < -params->l2_threshold, curr_grad > params->l2_threshold);
                reject_indices[i] = reject;
                remaining_indices[i] = constant_time::logical_not(reject);
            }

            int remaining_count = std::accumulate(remaining_indices.begin(), remaining_indices.end(), 0);


            /** first use as many "remaining" samples as possible */

            // generate random index permutation
            std::vector<int> permutation(dataset->length);
            std::iota(std::begin(permutation), std::end(permutation), 1);  // [1,2,3,...]
            sgx_vector_shuffle(permutation);

            // zero out all elements that are not part of the remaining array
            for(auto &elem : permutation){
                elem *= remaining_indices[elem-1];
            }
            std::transform(permutation.begin(), permutation.end(), permutation.begin(), [](int &c){return c-1;}); // make it 0-indexed

            // put corresponding rows into tree_indices
            int taken_rows = 0;
            for(int i=0; i<dataset->length; i++){
                bool use_row = constant_time::logical_and(permutation[i] != -1, taken_rows < number_of_rows);
                taken_rows += (int) use_row;
                // touch the entire tree_samples vector, to hide which one is added
                for(int j=0; j < dataset->length; j++){
                    tree_indices[j] = constant_time::select(constant_time::logical_and(j == permutation[i], use_row), 1, tree_indices[j]);
                }
            }

            /** if necessary, fill up with (randomly chosen and clipped) rejected samples */

            int num_additional_samples_required = constant_time::max(number_of_rows - remaining_count, 0);

            // generate random index permutation
            std::iota(std::begin(permutation), std::end(permutation), 1);  // [1,2,3,...]
            sgx_vector_shuffle(permutation);

            // zero out all elements that are not part of the rejected array
            for(auto &elem : permutation){
                elem *= reject_indices[elem-1];
            }
            std::transform(permutation.begin(), permutation.end(), permutation.begin(), [](int &c){return c-1;}); // make it 0-indexed

            // put corresponding rows into tree_indices
            taken_rows = 0;
            for(int i=0; i<dataset->length; i++) {
                // use row iff the row is part of "rejected_indices" and we still need more samples
                bool use_row = constant_time::logical_and(permutation[i] != -1, taken_rows < num_additional_samples_required);
                taken_rows += (int) use_row;
                // clip gradient if this row is used
                double clipped_gradient = clamp(dataset->gradients[i],-params->l2_threshold, params->l2_threshold);
                dataset->gradients[i] = constant_time::select(use_row, clipped_gradient, dataset->gradients[i]);
                // touch the entire tree_indices vector, to hide which one is added
                for(int j=0; j<dataset->length; j++){
                    // if we are at the right element, write 1, otherwise keep content
                    tree_indices[j] = constant_time::select(constant_time::logical_and(j == permutation[i], use_row), 1, tree_indices[j]);
                }
            }
            
        } else {
            // no GDF, just randomly select <number_of_rows> rows. This should not require hardening.
            // Note, this causes the leaves to be clipped after building the tree.
            vector<int> all_indices(dataset->length);
            std::iota(std::begin(all_indices), std::end(all_indices), 0);
            sgx_vector_shuffle(all_indices);
            for(int i=0; i<number_of_rows; i++){       
                tree_indices[all_indices[i]] = 1;
            }            
        }

        DataSet tree_dataset = dataset->get_subset(tree_indices);

        // build tree
        DPTree tree = DPTree(params, &tree_params, &tree_dataset, tree_index);
        tree.fit();
        trees.push_back(tree);

        // remove rows
        *dataset = dataset->remove_rows(tree_indices);

        // sgx_printf("Tree %i done. Instances left: %i\n", tree_index, dataset->length);
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
}
