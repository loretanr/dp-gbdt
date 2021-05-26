#include <string>
#include <numeric>
#include <algorithm>
#include <iostream>
#include "tree.h"
using namespace std;


namespace xgboost {
	Tree::Tree() :
		split_feature(0),
		split_value(0.0),
		split_gain(0.0),
		internal_value(0.0),
		leaf_value(0.0),
		tree_left(nullptr),
		tree_right(nullptr) {
	};
	//Tree::~Tree() {};

	//递归树的每一个分支，得到叶子节点值
	float Tree::PredictLeafValue(const vector<float>& dataset_one) {
		if (!this->tree_left && !this->tree_right) {
			return this->leaf_value;
		}
		else if (dataset_one[this->split_feature] <= this->split_value) {
			return this->tree_left->PredictLeafValue(dataset_one);
		}
		else {
			return this->tree_right->PredictLeafValue(dataset_one);
		}
	}
	
	//递归打印树结构（json形式）
	std::string Tree::DescribeTree() {
		if (!this->tree_left && !this->tree_right) {
			return "{\"leaf_value\":" + to_string(this->leaf_value) + "}";
		}

		std::string left_info = this->tree_left->DescribeTree();
		std::string right_info = this->tree_right->DescribeTree();
		std::string tree_structure;
		tree_structure = "{\"split_feature\":" + to_string(this->split_feature) + \
			",\"split_value\":" + to_string(this->split_value) + \
			",\"split_gain\":" + to_string(this->split_gain) + \
			",\"internal_value\":" + to_string(this->internal_value) + \
			",\"tree_left\":" + left_info + \
			",\"tree_right\":" + right_info + "}";
		return tree_structure;
	}
}
