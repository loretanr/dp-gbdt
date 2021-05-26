#include <limits>
#include <queue>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <thread>

#include "Dataset.h"
#include "Tree.h"
#include "utils.h"

#define FLOAT_MIN -1e10

Tree::Tree(Dataset* dataset, ModelParam params): dataset(dataset), params(params),
    categoric_attr_maps(dataset->categoric_attr_maps),
    categoric_attr_inverse_maps(dataset->categoric_attr_inverse_maps),
    label_maps(dataset->label_maps), label_inverse_maps(dataset->label_inverse_maps) {
    calculate_gh();
    root = new TreeNode();
    root->instances = dataset->instances("train");
    float G, H;
    G = accumulate(g.begin(), g.end(), 0.0);
    H = accumulate(h.begin(), h.end(), 0.0);
    root->weight = -G / (H + params.lambda);
}

Tree::~Tree() {
    queue<TreeNode*> node_que;
    node_que.push(root);
    while (!node_que.empty()) {
        TreeNode* node = node_que.front();
        if (node->left)
            node_que.push(node->left);
        if (node->right)
            node_que.push(node->right);
        delete node;
        node_que.pop();
    }
}

void Tree::grow_tree() {
    queue<TreeNode*> node_que;
    node_que.push(root);
    while (!node_que.empty() && node_que.front()->depth != params.max_depth) {
        TreeNode* node = node_que.front();
        if (!split_node(node)) {
            node_que.push(node->left);
            node_que.push(node->right);
        }
        node_que.pop();
    }
}

void Tree::prune_tree() {
    int num_leaves_pruned = 0;
    recursive_prune_tree(root, num_leaves_pruned);
    if (num_leaves_pruned != 0) {
        cout << "Prune tree happened: " << num_leaves_pruned << " leaves pruned..." << endl;
    } else {
        cout << "No leaves pruned during pruning tree period..." << endl;
    }
}

void Tree::inference_tree(float learning_rate) {
    for (int i = 0; i < dataset->num_instances(); ++i) {
        vector<float>& instance = dataset->get(i);
        float pred = inference_tree(instance);
        dataset->set_pred(i, pred*learning_rate);
    }
}

float Tree::inference_tree(vector<float>& instance) {
    TreeNode* node = root;
    while (!node->is_leaf()) {
        if (node->split_attr_type == 1) {
            if (instance[node->split_attr] <= node->split_value) {
                node = node->left;
            } else {
                node = node->right;
            }
        } else if (node->split_attr_type == 2) {
            if (instance[node->split_attr] == node->split_value) {
                node = node->left;
            } else {
                node = node->right;
            }
        }
    }
    return node->weight;
}

void Tree::print_tree() {
    recursive_print_tree(root);
}

void Tree::calculate_gh() {
    g = vector<float>(dataset->num_instances());
    h = vector<float>(dataset->num_instances());
    for (int i = 0; i < dataset->num_instances(); ++i) {
        g[i] = calculate_g(dataset->get_label(i), dataset->get_pred(i));
        h[i] = calculate_h(dataset->get_label(i), dataset->get_pred(i));
    }
}

void Tree::recursive_prune_tree(TreeNode* node, int& num_leaves_pruned) {
    if (node == nullptr || node->is_leaf()) {
        return;
    }
    recursive_prune_tree(node->left, num_leaves_pruned);
    recursive_prune_tree(node->right, num_leaves_pruned);
    if (node->split_gain < 0 && node->left->is_leaf() && node->right->is_leaf()) {
        delete node->left;
        delete node->right;
        node->left = nullptr;
        node->right = nullptr;
        num_leaves_pruned++;
    }
}

void Tree::recursive_print_tree(TreeNode* node) {
    if (node->is_leaf()) {
        return;
    }

    for (int i = 0; i < node->depth; ++i) {
        cout << ":  ";
    }
    if (node->split_attr_type == 1) {
        cout << "Attr" << node->split_attr << " <= " << node->split_value;
    } else if (node->split_attr_type == 2) {
        string split_value = categoric_attr_inverse_maps[node->split_attr][int(node->split_value)];
        cout << "Attr" << node->split_attr << " = " << split_value;
    }
    if (node->left->is_leaf()) {
        cout << " (" << node->left->weight << ")" << endl;
    } else {
        cout << endl;
    }
    recursive_print_tree(node->left);


    for (int i = 0; i < node->depth; ++i) {
        cout << ":  ";
    }
    if (node->split_attr_type == 1) {
        cout << "Attr" << node->split_attr << " > " << node->split_value;
    } else if (node->split_attr_type == 2) {
        string split_value = categoric_attr_inverse_maps[node->split_attr][int(node->split_value)];
        cout << "Attr" << node->split_attr << " != " << split_value;
    }
    if (node->right->is_leaf()) {
        cout << " (" << node->right->weight << ")" << endl;
    } else {
        cout << endl;
    }
    recursive_print_tree(node->right);
}

int Tree::split_node(TreeNode* node) {
    int best_attr, best_attr_type;
    float best_split_value;
    float max_gain;
    if (search_best_attr(node, best_attr, best_split_value, max_gain)) {
        return 1;
    }
    best_attr_type = dataset->get_attr_type(best_attr);

    float G_L = 0, G_R = 0, H_L = 0, H_R = 0;
    TreeNode* left = new TreeNode();
    TreeNode* right = new TreeNode();
    for (int i = 0; i < node->num_instances(); ++i) {
        int instance = node->get(i);
        if (best_attr_type == 1) {
            if (dataset->get(instance, best_attr) <= best_split_value) {
                left->add_instance(instance);
                G_L += g[instance];
                H_L += h[instance];
            } else {
                right->add_instance(instance);
                G_R += g[instance];
                H_R += h[instance];
            }
        } else if (best_attr_type == 2) {
            if (dataset->get(instance, best_attr) == best_split_value) {
                left->add_instance(instance);
                G_L += g[instance];
                H_L += h[instance];
            } else {
                right->add_instance(instance);
                G_R += g[instance];
                H_R += h[instance];
            }
        } else {
            cout << "attribute type invalid, check type..." << endl;
            exit(1);
        }
    }

    left->depth = node->depth + 1;
    left->loc = "left";
    left->weight = calculate_weight(G_L, H_L);
    right->depth = node->depth + 1;
    right->loc = "right";
    right->weight = calculate_weight(G_R, H_R);

    node->split_attr = best_attr;
    node->split_attr_type = best_attr_type;
    node->split_value = best_split_value;
    node->split_gain = max_gain;
    node->left = left;
    node->right = right;
    return 0;
}

int Tree::search_best_attr(TreeNode* node, int& best_attr, float& best_split_value,
                           float& max_gain) {
    max_gain = FLOAT_MIN;
    int m = node->num_instances();
    int n = dataset->num_attributes();
    int t = params.min_examples_leaf - 1;
    int s = m - params.min_examples_leaf;
    if ( t >= s) {
        return 1;
    }

    vector<int> attributes(n);
    for (int i = 0; i < n; ++i) {
        attributes[i] = i;
    }
    if (params.col_sample < 1.0) {
        shuffle_attributes(attributes);
    }

    int num_col_sample = min(max(1, int(params.col_sample * n)), n);
    if (params.multi_threads) {
        vector<int> split_attrs(attributes.begin(), attributes.begin()+num_col_sample);
        vector<float> split_values(num_col_sample);
        vector<float> split_gains(num_col_sample, FLOAT_MIN);
        vector<thread> threads;

        int num_threads = std::thread::hardware_concurrency() * 2;
        int num_patches = int(ceil(float(num_col_sample) / num_threads));
        for (int p = 0; p < num_patches; ++p) {
            threads.clear();
            for (int i = p * num_threads; i < min((p+1)*num_threads, num_col_sample); ++i) {
                threads.push_back(thread(&Tree::search_best_split, this, node, split_attrs[i], t, s,
                                         std::ref(split_values[i]), std::ref(split_gains[i])));
            }
            for (int i = 0; i < threads.size(); ++i) {
                threads[i].join();
            }
        }
        for (int i = 0; i < num_col_sample; ++i) {
            if (split_gains[i] > max_gain) {
                best_attr = split_attrs[i];
                best_split_value = split_values[i];
                max_gain = split_gains[i];
            }
        }
    } else {
        for (int i = 0; i < num_col_sample; ++i) {
            int attr = attributes[i];
            float split_value, gain;
            if (!search_best_split(node, attr, t, s, split_value, gain)) {
                if (gain > max_gain) {
                    best_attr = attr;
                    best_split_value = split_value;
                    max_gain = gain;
                }
            }
        }
    }
    if (max_gain == FLOAT_MIN) {
        return 1;
    }
    return 0;
}

int Tree::search_best_split(TreeNode* node, int attribute, int t, int s,
                             float& split_value, float& gain) {
    switch (dataset->get_attr_type(attribute)) {
        case 1:
            return search_best_numeric_split(node, attribute, t, s, split_value, gain);
        case 2:
            return search_best_categoric_split(node, attribute, t, s, split_value, gain);
        default:
            cout << "attribute type invalid, check type..." << endl;
            exit(1);
    }
}

int Tree::search_best_numeric_split(TreeNode* node, int attribute, int t, int s,
                                    float& split_value, float& gain) {
    vector<int> sorted_instances = node->sort_instances_by_attri(dataset, attribute);
    gain = FLOAT_MIN;

    float G_L = 0, G_R = 0;
    float H_L = 0, H_R = 0;
    for (int i = 0; i < node->num_instances(); ++i) {
        if (i <= t) {
            G_L += g[sorted_instances[i]];
            H_L += h[sorted_instances[i]];
        } else {
            G_R += g[sorted_instances[i]];
            H_R += h[sorted_instances[i]];
        }
    }

    while (t < s) {
        float value1 = dataset->get(sorted_instances[t], attribute);
        float value2 = dataset->get(sorted_instances[t+1], attribute);
        while (t < s && value1 == value2) {
            G_L += g[sorted_instances[t+1]];
            G_R -= g[sorted_instances[t+1]];
            H_L += h[sorted_instances[t+1]];
            H_R -= h[sorted_instances[t+1]];
            t++;
            if (t >= s) {
                break;
            }
            value2 = dataset->get(sorted_instances[t+1], attribute);
        }
        if (t >= s) {
            break;
        }
        float tmp_gain = calculate_gain(G_L, G_R, H_L, H_R);
        if (tmp_gain > gain) {
            gain = tmp_gain;
            split_value = dataset->get(sorted_instances[t], attribute);
        }
        G_L += g[sorted_instances[t+1]];
        G_R -= g[sorted_instances[t+1]];
        H_L += h[sorted_instances[t+1]];
        H_R -= h[sorted_instances[t+1]];
        t++;
    }
    if (gain == FLOAT_MIN) {
        return 1;
    }
    return 0;
}

int Tree::search_best_categoric_split(TreeNode* node, int attribute, int t, int s,
                                      float& split_value, float& gain) {
    gain = FLOAT_MIN;
    unordered_map<int, pair<int, float>> Gs, Hs;
    float G = 0, H = 0;
    for (int i = 0; i < node->num_instances(); ++i) {
        int instance = node->get(i);
        G += g[instance];
        H += h[instance];
        int attr_value = int(dataset->get(instance, attribute));
        if (Gs.find(attr_value) == Gs.end()) {
            Gs[attr_value] = make_pair(1, g[instance]);
        } else {
            Gs[attr_value].first++;
            Gs[attr_value].second += g[instance];
        }
        if (Hs.find(attr_value) == Hs.end()) {
            Hs[attr_value] = make_pair(1, h[instance]);
        } else {
            Hs[attr_value].first++;
            Hs[attr_value].second += h[instance];
        }
    }
    if (Gs.size() == 1) {
        return 1;
    }

    float G_L, G_R, H_L, H_R;
    for (auto attr_value: Gs) {
        if (attr_value.second.first < params.min_examples_leaf) {
            continue;
        }
        if (node->num_instances() - attr_value.second.first < params.min_examples_leaf) {
            continue;
        }
        G_L = attr_value.second.second; G_R = G - G_L;
        H_L = Hs[attr_value.first].second; H_R = H - H_L;
        float tmp_gain = calculate_gain(G_L, G_R, H_L, H_R);
        if (tmp_gain > gain) {
            gain = tmp_gain;
            split_value = attr_value.first;
        }
    }
    if (gain == FLOAT_MIN) {
        return 1;
    }
    return 0;
}

float Tree::calculate_g(float y, float y_pred) {
    if (params.objective == "regression") {
        return square_first_gradient(y, y_pred);
    } else if (params.objective == "binary") {
        return logistic_first_gradient(y, y_pred);
    } else {
        cout << "unknown objective type: " << params.objective << endl;
        exit(1);
    }
}

float Tree::calculate_h(float y, float y_pred) {
    if (params.objective == "regression") {
        return square_second_gradient(y, y_pred);
    } else if (params.objective == "binary") {
        return logistic_second_gradient(y, y_pred);
    } else {
        cout << "unknown objective type: " << params.objective << endl;
        exit(1);
    }
}

float Tree::calculate_gain(float G_L, float G_R, float H_L, float H_R) {
    float tmp1 = G_L * G_L / (H_L + params.lambda);
    float tmp2 = G_R * G_R / (H_R + params.lambda);
    float tmp3 = (G_L + G_R) * (G_L + G_R) / (H_L + H_R + params.lambda);
    return (tmp1 + tmp2 - tmp3) / 2 - params.gamma;
}

float Tree::calculate_weight(float G, float H) {
    float tmp = H + params.lambda;
    if (tmp == 0) {
        return numeric_limits<float>::min();
    }
    return -G / tmp;
}
