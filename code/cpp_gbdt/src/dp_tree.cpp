#include "dp_tree.h"

//#include <quadmath.h>

 //: tree_index(tree_index), learning_rate(learning_rate), l2_threshold(l2_threshold), l2_lambda(l2_lambda), privacy_budget(privacy_budget), delta_g(delta_g), delta_v(delta_v), loss(loss), max_depth(max_depth), max_leaves(max_leaves), min_samples_split(min_samples_split), leaf_clipping(leaf_clipping), use_bfs(use_bfs), use_3_trees(use_3_trees), use_decay(use_decay), cat_idx(cat_idx), num_idx(num_idx)
DPTree::DPTree(ModelParams *params, TreeParams *tree_params, DataSet *dataset, size_t tree_index): 
    params(params),
    tree_params(tree_params), 
    dataset(dataset),
    tree_index(tree_index)
{
   /*  // create a matrix whose rows contain the columns of X, but without duplicates
    for (int i=0; i<dataset->num_x_cols; i++){
        X_unique.push_back(set<double>());
    }
    for (int row=0; row < dataset->length; row++) {
        for (int col=0; col < dataset->num_x_cols; col++) {
            X_unique[col].insert(dataset->X[row][col]);
        }
    } */
    //nodes = new vector<TreeNode>;
}


// TODO destroy the nodes, done in ensemble for now.
DPTree::~DPTree() {}


// Fit the tree to the data
void DPTree::fit()
{
    if (params->use_dfs) {
        // keep track which samples will be available in a node for spliting
        vector<int> live_samples(dataset->length);
        std::iota(std::begin(live_samples), std::end(live_samples), 0);

        root_node = make_tree_DFS(0, live_samples);
        //nodes = collect_nodes(*root_node);
    } else {
        throw runtime_error("non-DFS not yet implemented.");
    }

    // leaf clipping TODO
    if (params->leaf_clipping) {
        double threshold = this->params->l2_threshold * std::pow((1-this->params->learning_rate), this->tree_index);
        for (auto &leaf : this->leaves) {
            leaf->prediction = clip(leaf->prediction, -1 * threshold, threshold);
        }
    }

    // add noise to predictions TODO
    double privacy_budget_for_leaf_nodes = tree_params->tree_privacy_budget  / 2;
    double laplace_scale = tree_params->delta_v / privacy_budget_for_leaf_nodes;
    add_laplacian_noise(this->leaves, laplace_scale);
    
}


// Recursively build tree, DFS approach, 1st instance returns root node
TreeNode *DPTree::make_tree_DFS(int current_depth, vector<int> live_samples)
{
    // max depth reached or not enough samples -> leaf node
    if ( (current_depth == params->max_depth) or
        live_samples.size() < (size_t) params->min_samples_split) {
            TreeNode *leaf = make_leaf_node(current_depth, live_samples);
            LOG_DEBUG("max_depth ({1}) or min_samples ({2})-> leaf (pred={3:.2f})", current_depth, live_samples.size(), leaf->prediction);
            return leaf;
        }

    // only use the samples that actually end up in this node
    // note that the cols of X are rows in X_live
    VVD X_live;
    vector<double> gradients_live;
    for(int col=0; col < dataset->num_x_cols; col++) {
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

    vector<bool> lhs;
    samples_left_right_partition(lhs, X_live, gradients_live, node->split_attr, node->split_value);
    vector<int> left_live_samples, right_live_samples;
    for (size_t i=0; i<live_samples.size(); i++) {
        if (lhs[i]) {
            left_live_samples.push_back(live_samples[i]);
        } else {
            right_live_samples.push_back(live_samples[i]);
        }
    }
    LOG_DEBUG("best split @ {1}, val {2:.2f}, gain {3:.5f}, curr_depth {4}, samples {5} ->({6},{7})", 
        node->split_attr, node->split_value, node->split_gain, current_depth, 
        live_samples.size(), left_live_samples.size(), right_live_samples.size()); 

    node->left = make_tree_DFS(current_depth + 1, left_live_samples);
    node->right = make_tree_DFS(current_depth + 1, right_live_samples);

    //(*nodes).push_back(node);
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
    leaf->prediction = compute_prediction(gradients, y);
    leaves.push_back(leaf); // also some general node list?
    return(leaf);
}


vector<TreeNode> DPTree::collect_nodes(TreeNode rootnode)
{
    // dummy return val
    vector<TreeNode> bla;
    return bla;
}


double DPTree::compute_prediction(vector<double> gradients, vector<double> y)
{
    double prediction = (-1 * std::accumulate(gradients.begin(), gradients.end(), 0.0)
                            / (gradients.size() + this->params->l2_lambda));
    return prediction;
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


//Find best split of data using the exponential mechanism
TreeNode *DPTree::find_best_split(VVD &X_live, vector<double> &gradients_live, int current_depth)
{
    double privacy_budget_for_node;
    if ((current_depth != 0) and params->use_decay) {
        privacy_budget_for_node = (tree_params->tree_privacy_budget) / 2 / pow(2, current_depth);
    } else {
        privacy_budget_for_node = (tree_params->tree_privacy_budget)/2/params->max_depth;
    }
    if (params->use_3_trees and (current_depth != 0)) {
        // Except for the root node, budget is divided by the 3-nodes
        privacy_budget_for_node /= 2;
    }

    // LOG_DEBUG("Using {1:.4f} budget for internal leaf nodes", privacy_budget_for_node);

    vector<SplitCandidate> probabilities;
    double max_gain = numeric_limits<double>::min();
    
    
    // iterate over features
    for (int feature_index=0; feature_index < dataset->num_x_cols; feature_index++) {
        for (double feature_value : X_live[feature_index]) { // TODO, don't iterate over duplicates in X_live
            // compute gain
            double gain = compute_gain(X_live, gradients_live, feature_index, feature_value);
            // feature cannot be chosen, skipping
            if (gain == -1) {
                continue;
            }
            gain = (privacy_budget_for_node * gain) / (2 * tree_params->delta_g);
            max_gain = (gain > max_gain) ? gain : max_gain; // unused
            SplitCandidate candidate = SplitCandidate(feature_index, feature_value, gain);
            probabilities.push_back(candidate);
        }
    }
    int index = exponential_mechanism(probabilities, max_gain);
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
    }
    node->depth = current_depth;
    return node;
}

// This gain is the simplified formula for least squares loss function
double DPTree::compute_gain(VVD &samples, vector<double> &gradients_live, int feature_index, double feature_value)
{
    // partition into lhs / rhs
    vector<bool> lhs;
    samples_left_right_partition(lhs, samples, gradients_live, feature_index, feature_value);

    int lhs_size = std::count(lhs.begin(), lhs.end(), true);
    int rhs_size = std::count(lhs.begin(), lhs.end(), false);

    // if all samples go on the same side it's useless to split on this value
    if ( lhs_size == 0 or rhs_size == 0 ) {
        return -1;
    }

    double lhs_gain = 0, rhs_gain = 0;
    for (size_t index=0; index<lhs.size(); index++) {
        lhs_gain += (unsigned) lhs[index] * (gradients_live)[index];
        rhs_gain += (unsigned) (not lhs[index]) * (gradients_live)[index];
    }
    lhs_gain = std::pow(lhs_gain,2) / (lhs_size + params->l2_lambda);
    rhs_gain = std::pow(rhs_gain,2) / (rhs_size + params->l2_lambda);

    double total_gain = lhs_gain + rhs_gain;
    total_gain = std::floor(total_gain * 1e10) / 1e10;

    return std::max(total_gain, 0.0);
}


// the result is a bool array that will indicate left/right
void DPTree::samples_left_right_partition(vector<bool> &lhs, VVD &samples, vector<double> &gradients_live, int feature_index, double feature_value)
{
    // if the feature is categorical
    if(std::find((params->cat_idx).begin(), (params->cat_idx).end(), feature_index) != (params->cat_idx).end()) {
        for (auto sample : samples[feature_index]) {
            lhs.push_back(sample == feature_value);
        }
    } else { // feature is numerical
        for (auto sample : samples[feature_index]) {
            lhs.push_back(sample < feature_value);
        }
    }
}


int DPTree::exponential_mechanism(vector<SplitCandidate> &probs, double max_gain)
{
    int count = std::count_if(probs.begin(), probs.end(),[](SplitCandidate c){ return c.gain > 0; });

    if (count == 0) {
        return -1;
    }

    // TODO need quadmath for overflows?

    vector<double> gains, probabilities, partials(probs.size());

    for (auto p : probs) {
        if (p.gain != 0) {
            gains.push_back(p.gain);
        }
    }
    for (auto &prob : probs) {
        if (prob.gain <= 0) {
            probabilities.push_back(0);   
        } else {
            probabilities.push_back( exp( (double) prob.gain - log_sum_exp(gains)) );
        }
    }

    // all values in [0,1]
    std::partial_sum(probabilities.begin(), probabilities.end(), partials.begin());

    srand(time(0));
    double rand01 = ((double) std::rand() / (RAND_MAX));

    // disable randomness for debug
    if (not RANDOMIZATION) {
        auto max_elem = std::max_element(probabilities.begin(), probabilities.end());
        return std::distance(probabilities.begin(), max_elem);
    }
    // try to find a candidate at least 10 times before giving up and making the node a leaf node
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

void DPTree::add_laplacian_noise(vector<TreeNode *> leaves, double laplace_scale)
{
    LOG_INFO("Adding Laplace noise to leaves (Scale {1:.2f})", laplace_scale);
    srand(time(NULL));
    Laplace lap(laplace_scale, rand());
    for (auto leaf : leaves) {
        double noise = 0;
        if (RANDOMIZATION) {
            noise = lap.return_a_random_variable(laplace_scale);
        }
        LOG_DEBUG("({1:.3f} -> {2:.3f})", leaf->prediction, leaf->prediction+noise);
        leaf->prediction += noise;
    }
    // debugging TODO REMOVE
    double sum = 0;
    for (auto leaf : leaves) {
        sum += leaf->prediction;
    }
    LOG_INFO("LEAFSUM {1:.8f}", sum);
    if(VERIFICATION_MODE) {
        VERIFICATION_LOG("LEAFVALUESSUM {0:.10f}", sum);
    }
}


void DPTree::recursive_print_tree(TreeNode* node) {

    if (node->is_leaf()) {
        return;
    }
    // check if split uses categorical attr
    bool categorical = std::find( ((*(this->params)).cat_idx).begin(), ((*(this->params)).cat_idx).end(), node->split_attr) != ((*(this->params)).cat_idx).end();
    
    if (categorical) {
        std::cout << std::defaultfloat;
    } else {
        std::cout << std::fixed;
    }

    for (int i = 0; i < node->depth; ++i) { cout << ":  "; }

    if (!categorical) {
        cout << "Attr" << std::setprecision(3) << node->split_attr << " < " << std::setprecision(3) << node->split_value;
    } else { // else if (node->split_attr == 2)
        double split_value = (node->split_value); // categorical, hacked
        cout << "Attr" << node->split_attr << " = " << split_value;
    }
    if (node->left->is_leaf()) {
        cout << " (" << "L-leaf" << ")" << endl; // node->left->weight
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
        cout << "Attr" << std::setprecision(3) << node->split_attr << " >= " << std::setprecision(3) << node->split_value;
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


void DPTree::delete_tree(TreeNode *node)
{
    if (not node->is_leaf()) {
        delete_tree(node->left);
        delete_tree(node->right);
    }
    delete node;
    return;
}
