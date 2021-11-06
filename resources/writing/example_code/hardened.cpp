


// TODO is this the best example I can get?



// recursively walk through decision tree
double predict(vector<double> row, TreeNode *node)
{
    // determine whether the current node splits on a categorical or numerical feature
    bool categorical = false;
    for(auto cat_feature : params->cat_idx){
        // touch the entire array
        categorical = constant_time::logical_or(categorical, cat_feature == node->split_attr);
    }

    // always go down to max_depth
    if(node->depth < params->max_depth){
        double row_val = row[node->split_attr];

        // We hide the real path a sample row takes, -> go down both paths at every decision node.
        // Further we hide whether the current node splits on a categorical/numerical feature. 
        double left_result = predict(row, node->left);
        double right_result = predict(row, node->right);

        // decide whether we take the value from the left or right child
        double next_levelprediction = constant_time::select(categorical,
            constant_time::select((row_val == node->split_value), left_result, right_result),
            constant_time::select((row_val < node->split_value), left_result, right_result) );
    }

    // if we are a leaf, we take the leaf value, otherwise we take the value of the child node.
    return constant_time::select(node->is_leaf, node->prediction, next_levelprediction);
}


// recursively walk through decision tree
double DPTree::predict(vector<double> row, TreeNode *node)
{
    // recursion base case
    if(node->is_leaf()){
        return node->prediction;
    }

    double row_val = row[node->split_attr];

    if (std::find((params->cat_idx).begin(), (params->cat_idx).end(), node->split_attr) != (params->cat_idx).end()) {
        // categorical feature
        if (row_val == node->split_value){
            return predict(row, node->left);
        }
    } else { // numerical feature
        if (row_val < node->split_value){
            return predict(row, node->left);
        }
    }
    return predict(row, node->right);
}