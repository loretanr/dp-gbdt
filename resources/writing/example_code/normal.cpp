// recursively walk through the decision tree
double predict(vector<double> sample_row, TreeNode *node)
{
    // base case
    if(node->is_leaf()){
        return node->prediction;
    }

    double sample_value = sample_row[node->split_attr];
    
    // recurse left or right
    if (sample_value < node->split_value){
        return predict(sample_row, node->left);
    }
    return predict(sample_row, node->right);
}