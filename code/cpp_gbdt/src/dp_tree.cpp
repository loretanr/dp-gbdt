#include <numeric>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "dp_tree.h"
#include "laplace.h"
#include "logging.h"
#include "spdlog/spdlog.h"

extern std::ofstream verification_logfile;
extern bool VERIFICATION_MODE;

using namespace std;


/** Constructors */

DPTree::DPTree(ModelParams *params, TreeParams *tree_params, DataSet *dataset, size_t tree_index): 
    params(params),
    tree_params(tree_params), 
    dataset(dataset),
    tree_index(tree_index) {}

DPTree::~DPTree() {}


/** Methods */

// Fit the tree to the data
void DPTree::fit()
{
    if (params->use_dfs) {
        // keep track which samples will be available in a node for spliting
        vector<int> live_samples(dataset->length);
        std::iota(std::begin(live_samples), std::end(live_samples), 0);

        root_node = make_tree_DFS(0, live_samples);
    } else {
        throw runtime_error("non-DFS not yet implemented.");
    }

    // leaf clipping
    if (params->leaf_clipping) {
        double threshold = this->params->l2_threshold * std::pow((1-this->params->learning_rate), this->tree_index);
        for (auto &leaf : this->leaves) {
            leaf->prediction = clamp(leaf->prediction, -1 * threshold, threshold);
        }
    }

    // add laplace noise to predictions
    double privacy_budget_for_leaf_nodes = tree_params->tree_privacy_budget  / 2;
    double laplace_scale = tree_params->delta_v / privacy_budget_for_leaf_nodes;
    add_laplacian_noise(laplace_scale);
}


// Recursively build tree, DFS approach, first instance returns root node
TreeNode *DPTree::make_tree_DFS(int current_depth, vector<int> live_samples)
{
    // max depth reached or not enough samples -> leaf node
    if ( (current_depth == params->max_depth) or 
            live_samples.size() < (size_t) params->min_samples_split) {
        TreeNode *leaf = make_leaf_node(current_depth, live_samples);
        LOG_DEBUG("max_depth ({1}) or min_samples ({2})-> leaf (pred={3:.2f})",
            current_depth, live_samples.size(), leaf->prediction);
        return leaf;
    }

    // only use the samples that actually end up in this node
    // note that the cols of X are rows in X_live
    VVD X_live; vector<double> gradients_live;
    for(int col=0; col < dataset->num_x_cols; col++) {      // TODO ugly
        vector<double> temp;    
        for (auto elem : live_samples) {
            temp.push_back((dataset->X)[elem][col]);
        }
        X_live.push_back(temp);
    }
    for(auto elem : live_samples) {
        gradients_live.push_back((dataset->gradients)[elem]);
    }

    // find best split
    TreeNode *node = find_best_split(X_live, gradients_live, current_depth);

    // no split found
    if (node->is_leaf()) {
        LOG_DEBUG("no split found -> leaf");
        return node;
    }

    LOG_DEBUG("best split @ {1}, val {2:.2f}, gain {3:.5f}, curr_depth {4}, samples {5} ->({6},{7})", 
        node->split_attr, node->split_value, node->split_gain, current_depth, 
        node->lhs_size + node->rhs_size, node->lhs_size, node->rhs_size);

    // Update live samples and continue recursion
    vector<int> lhs;
    bool categorical = std::find((params->cat_idx).begin(), (params->cat_idx).end(), node->split_attr) != (params->cat_idx).end();
    samples_left_right_partition(lhs, X_live, gradients_live, node->split_attr, node->split_value, categorical);
    vector<int> left_live_samples, right_live_samples;
    for (size_t i=0; i<live_samples.size(); i++) {
        if (lhs[i]) {
            left_live_samples.push_back(live_samples[i]);
        } else {
            right_live_samples.push_back(live_samples[i]);
        }
    }

    node->left = make_tree_DFS(current_depth + 1, left_live_samples);
    node->right = make_tree_DFS(current_depth + 1, right_live_samples);

    return node;
}


TreeNode *DPTree::make_leaf_node(int current_depth, vector<int> &live_samples)
{
    TreeNode *leaf = new TreeNode(true);
    leaf->depth = current_depth;

    vector<double> y, gradients;
    for (auto index : live_samples) {
        y.push_back((this->dataset->y)[index]);
        gradients.push_back(this->dataset->gradients[index]);
    }
    // compute prediction
    leaf->prediction = (-1 * std::accumulate(gradients.begin(), gradients.end(), 0.0)
                            / (gradients.size() + this->params->l2_lambda));
    this->leaves.push_back(leaf);
    return(leaf);
}


vector<double> DPTree::predict(VVD &X)
{
    vector<double> predictions;
    for (auto row : X) {
        double pred = _predict(&row, this->root_node);
        predictions.push_back(pred);
    }

    return predictions;
}

// recursively walk through decision tree
double DPTree::_predict(vector<double> *row, TreeNode *node)
{
    if(node->is_leaf()){
        return node->prediction;
    }
    double row_val = (*row)[node->split_attr];

    if (std::find((params->cat_idx).begin(), (params->cat_idx).end(), node->split_attr) != (params->cat_idx).end()) {
        // categorical feature
        if (row_val == node->split_value){
            return _predict(row, node->left);
        }
    } else {
        if (row_val < node->split_value){               // have LEQ here, theo has LT (but should only
            return _predict(row, node->left);            // rarely affect debugging determinism)
        }
    }
    
    return _predict(row, node->right);
}


// Find best split of data using the exponential mechanism
TreeNode *DPTree::find_best_split(VVD &X_live, vector<double> &gradients_live, int current_depth)
{
    double privacy_budget_for_node;
    if ((current_depth != 0) and params->use_decay) {
        privacy_budget_for_node = (tree_params->tree_privacy_budget) / 2 / pow(2, current_depth);
    } else {
        privacy_budget_for_node = (tree_params->tree_privacy_budget) / 2 / params->max_depth;
    }

    vector<SplitCandidate> probabilities;
    double max_gain = numeric_limits<double>::min();
    int lhs_size;
    
    // iterate over features
    for (int feature_index=0; feature_index < dataset->num_x_cols; feature_index++) {
        bool categorical = std::find((params->cat_idx).begin(), (params->cat_idx).end(), feature_index) != (params->cat_idx).end();
        for (double feature_value : X_live[feature_index]) { // TODO, don't iterate over duplicates in X_live
            // compute gain
            double gain = compute_gain(X_live, gradients_live, feature_index, feature_value, lhs_size, categorical);
            // feature cannot be chosen, skipping
            if (gain == -1) {
                continue;
            }
            gain = (privacy_budget_for_node * gain) / (2 * tree_params->delta_g);
            max_gain = (gain > max_gain) ? gain : max_gain; // TODO unused ?
            SplitCandidate candidate = SplitCandidate(feature_index, feature_value, gain);
            candidate.lhs_size = lhs_size;
            candidate.rhs_size = gradients_live.size() - lhs_size;
            probabilities.push_back(candidate);
        }
    }

    // choose a split using the exponential mechanism
    int index = exponential_mechanism(probabilities, max_gain);

    // construct the node
    TreeNode *node;
    if (index == -1) {
        node = new TreeNode(true);
        node->left = nullptr;
        node->right = nullptr;
    } else {
        node = new TreeNode(false);
        node->split_attr = probabilities[index].feature_index;
        node->split_value = probabilities[index].split_value;
        node->split_gain = probabilities[index].gain;
        node->lhs_size = probabilities[index].lhs_size;
        node->rhs_size = probabilities[index].rhs_size;
    }
    node->depth = current_depth;
    return node;
}


// This gain is the simplified formula for least squares loss function // TODO adapt for classification
double DPTree::compute_gain(VVD &samples, vector<double> &gradients_live,
    int feature_index, double feature_value, int &lhs_size, bool categorical)
{
    // partition into lhs / rhs
    vector<int> lhs;
    samples_left_right_partition(lhs, samples, gradients_live, feature_index, feature_value, categorical);

    int _lhs_size = std::count(lhs.begin(), lhs.end(), 1);
    int _rhs_size = std::count(lhs.begin(), lhs.end(), 0);

    lhs_size = _lhs_size;

    // if all samples go on the same side it's useless to split on this value
    if ( _lhs_size == 0 or _rhs_size == 0 ) {
        return -1;
    }

    double lhs_gain = 0, rhs_gain = 0;
    for (size_t index=0; index<lhs.size(); index++) {
        lhs_gain += lhs[index] * (gradients_live)[index];
        rhs_gain += (not lhs[index]) * (gradients_live)[index];
    }
    lhs_gain = std::pow(lhs_gain,2) / (_lhs_size + params->l2_lambda);
    rhs_gain = std::pow(rhs_gain,2) / (_rhs_size + params->l2_lambda);

    double total_gain = lhs_gain + rhs_gain;
    total_gain = std::floor(total_gain * 1e10) / 1e10;

    return std::max(total_gain, 0.0);
}


// the result is a bool array that will indicate left/right
void DPTree::samples_left_right_partition(vector<int> &lhs, VVD &samples, vector<double> &gradients_live,
            int feature_index, double feature_value, bool categorical)
{
    // if the feature is categorical
    if(categorical) {
        for (auto sample : samples[feature_index]) {
            size_t value = sample == feature_value;
            lhs.push_back(value);
        }
    } else { // feature is numerical
        for (auto sample : samples[feature_index]) {
            size_t value = sample < feature_value;
            lhs.push_back(value);
        }
    }
}


// Computes probabilities from the gains. (Larger gain -> larger probability to 
// be chosen for split). Then a cumulative distribution function is created from
// these probabilities. Then we can sample from it using a RNG.
// Returns the index of the chosen split.
int DPTree::exponential_mechanism(vector<SplitCandidate> &probs, double max_gain)
{
    // if no split has a positive gain, return. Node will become a leaf
    int count = std::count_if(probs.begin(), probs.end(),
        [](SplitCandidate c){ return c.gain > 0; });
    if (count == 0) {
        return -1;
    }

    // calculate the probabilities from the gains
    vector<double> gains, probabilities, partials(probs.size());
    for (auto p : probs) {
        gains.push_back(p.gain);
    }
    double lse = log_sum_exp(gains);
    for (auto prob : probs) {
        if (prob.gain <= 0) {
            probabilities.push_back(0);   
        } else {
            probabilities.push_back( exp(prob.gain - lse) );
        }
    }

    // non-dp: just choose largest probability
    if (!this->params->use_dp) {
        auto max_elem = std::max_element(probabilities.begin(), probabilities.end());
        // return index of the max_elem
        return std::distance(probabilities.begin(), max_elem);
    }

    // create a cumulative distribution function from the probabilities.
    // all values will be in [0,1]
    std::partial_sum(probabilities.begin(), probabilities.end(), partials.begin());

    double rand01 = ((double) std::rand() / (RAND_MAX));

    // try to find a candidate at least 10 times before giving up and making the node a leaf node
    // taken from Theos python code
    for (int tries=0; tries<10; tries++) {
        for (size_t index=0; index<partials.size(); index++) {
            if (partials[index] >= rand01) {
                return index;
            }
        }
        rand01 = ((double) rand() / (RAND_MAX));
    }
    return -1;
}


void DPTree::add_laplacian_noise(double laplace_scale)
{
    LOG_INFO("Adding Laplace noise to leaves (Scale {1:.2f})", laplace_scale);

    Laplace lap(laplace_scale, rand());

    // add noise from laplace distribution to leaves
    for (auto &leaf : this->leaves) {
        double noise = 0;

        // but only in dp mode
        if (this->params->use_dp) {
            noise = lap.return_a_random_variable(laplace_scale);
        }

        leaf->prediction += noise;

        LOG_DEBUG("({1:.3f} -> {2:.8f})", leaf->prediction, leaf->prediction+noise);
    }

    // rest is just for validation
    double sum = 0;
    for (auto leaf : this->leaves) {
        sum += leaf->prediction;
    }
    LOG_INFO("NUMLEAVES {1} LEAFSUM {2:.8f}", this->leaves.size(), sum);
    if(VERIFICATION_MODE) {
        VERIFICATION_LOG("LEAFVALUESSUM {0:.10f}", sum);
    }
}


// active in debug mode, prints the tree to console
void DPTree::recursive_print_tree(TreeNode* node) {

    if (node->is_leaf()) {
        return;
    }
    // check if split uses categorical attr
    bool categorical = std::find( ((*(this->params)).cat_idx).begin(),
        ((*(this->params)).cat_idx).end(), node->split_attr) != ((*(this->params)).cat_idx).end();
    
    if (categorical) {
        std::cout << std::defaultfloat;
    } else {
        std::cout << std::fixed;
    }

    for (int i = 0; i < node->depth; ++i) { cout << ":  "; }

    if (!categorical) {
        cout << "Attr" << std::setprecision(3) << node->split_attr << 
            " < " << std::setprecision(3) << node->split_value;
    } else {
        double split_value = (node->split_value); // categorical, hacked
        cout << "Attr" << node->split_attr << " = " << split_value;
    }
    if (node->left->is_leaf()) {
        cout << " (" << "L-leaf" << ")" << endl;
    } else {
        cout << endl;
    }

    recursive_print_tree(node->left);

    if (categorical) {
        std::cout << std::defaultfloat;
    } else {
        std::cout << std::fixed;
    }

    for (int i = 0; i < node->depth; ++i) { cout << ":  "; }
    if (!categorical) {
        cout << "Attr" << std::setprecision(3) << node->split_attr <<
            " >= " << std::setprecision(3) << node->split_value;
    } else {
        double split_value = node->split_value;
        cout << "Attr" << node->split_attr << " != " << split_value;
    }
    if (node->right->is_leaf()) {
        cout << " (" << "R-leaf" << ")" << endl;
    } else {
        cout << endl;
    }
    recursive_print_tree(node->right);
}


// free allocated ressources
void DPTree::delete_tree(TreeNode *node)
{
    if (not node->is_leaf()) {
        delete_tree(node->left);
        delete_tree(node->right);
    }
    delete node;
    return;
}