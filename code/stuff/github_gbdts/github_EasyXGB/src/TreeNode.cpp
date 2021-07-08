#include "TreeNode.h"

TreeNode::TreeNode(): left(nullptr), right(nullptr), depth(0), loc("root"),
                      split_attr(-1), split_value(0), split_gain(0) {
}

void TreeNode::add_instance(int id) {
    instances.push_back(id);
}

vector<int> TreeNode::sort_instances_by_attri(Dataset* dataset, int attribute) {
    vector<int> tmp = instances;
    sort(tmp.begin(), tmp.end(),
         [dataset, attribute](int a, int b){return dataset->get(a, attribute) < dataset->get(b, attribute);});
    return tmp;
}

int TreeNode::num_instances() {
    return instances.size();
}

int TreeNode::get(int index) {
    return instances[index];
}

bool TreeNode::is_leaf() {
    return left == nullptr && right == nullptr;
}

void TreeNode::print_node_info() {
    cout << endl;
    cout << "num instances: " << num_instances() << endl;
    cout << "split attr: " << split_attr << endl;
    cout << "split value: " << split_value << endl;
    cout << "split gain: " << split_gain << endl;
    cout << "weight: " << weight << endl;
    cout << "loc: " << loc << endl;
    cout << "depth: " << depth << endl;
}
