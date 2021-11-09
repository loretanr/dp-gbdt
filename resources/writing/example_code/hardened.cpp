// recursively walk through the decision tree
double predict(vector<double> sample_row, TreeNode *node)
{
    // always go down to max_depth
    if(node->depth < max_depth){
        
        double sample_value = sample_row[node->split_attr];

        // hide the real path a sample takes, -> go down both paths at every decision node.
        double left_result = predict(sample_row, node->left);
        double right_result = predict(sample_row, node->right);

        // decide whether we take the value from the left or right child
        bool is_smaller = constant_time::smaller(sample_value, node->split_value);
        double child_value = constant_time::select(is_smaller, left_result, right_result);
    }
    // if we are a leaf, take own value, otherwise we take the child's value.
    return constant_time::select(node->is_leaf, node->prediction, child_value);
}