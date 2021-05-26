#include <numeric>
#include <algorithm>
#include "xgboost.h"
#include "decision_tree.h"
#include "tree.h"
#include <sstream>
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
using namespace std;
using namespace rapidjson;


namespace xgboost {
	XGBoost::XGBoost(Config conf) :config(conf) {};
	XGBoost::~XGBoost() {};

	//训练模型主函数入口
	void XGBoost::fit(const vector<vector<float>>& features, const vector<int>& labels) {
		float mean = accumulate(labels.begin(), labels.end(), 0) / (float)labels.size();
		pred_0 = 0.5 * log((1 + mean) / (1 - mean));

		Gradients gradients;
		for (size_t i = 0; i < labels.size(); ++i) {
			gradients = CalculateGradHess(labels[i], pred_0);
			grad.push_back(gradients.grad);
			hess.push_back(gradients.hess);
		}

		for (int stage = 1; stage <= config.n_estimators; ++stage) {
			cout << "=============================== iter: " << stage << " ===============================" << endl;
			Tree *tree_stage;
			BaseDecisionTree base_decision_tree = BaseDecisionTree(config);
			tree_stage = base_decision_tree.fit(features, labels, grad, hess);
			trees.push_back(tree_stage);
			cout << tree_stage->DescribeTree() << endl;

			for (size_t i = 0; i < labels.size(); ++i) {
				float y_pred = tree_stage->PredictLeafValue(features[i]);
				gradients = CalculateGradHess(labels[i], y_pred);
				grad[i] += config.learning_rate * gradients.grad;
				hess[i] += config.learning_rate * gradients.hess;
			}
		}
	}

	//计算一阶和二阶导数
	Gradients XGBoost::CalculateGradHess(int y, float y_pred) {
		Gradients gradients;
		float pred = 1.0 / (1.0 + exp(-y_pred));
		float grad = (-y + (1 - y) * exp(pred)) / (1 + exp(pred));
		float hess = exp(pred) / pow((1 + exp(pred)), 2);
		gradients = { grad, hess };
		return gradients;
	}

	//给定样本特征，预测p值
	vector<float> XGBoost::PredictProba(const vector<float>& features) {
		float pred = pred_0;
		vector<float> res;
		float p_0;
		for (Tree *tree : trees) {
			pred += config.learning_rate * tree->PredictLeafValue(features);

		}
		p_0 = 1.0 / (1 + exp(2 * pred));
		res.push_back(p_0);
		res.push_back(1 - p_0);
		return res;
	}

	//模型保存为string
	std::string XGBoost::SaveModelToString() {
		std::string ss;
		//Trees
		ss += "{\"Trees\":[";
		for (size_t i = 0; i < trees.size(); ++i) {
			Tree *tree = trees[i];
			ss += trees[i]->DescribeTree();
			ss += ",";
		}
		ss = ss.substr(0, ss.length() - 1) + "]";
		
		//Param
		ss += ",\"Param\":{";
		ss = ss + "\"n_estimators\":" + to_string(config.n_estimators) + ",";
		ss = ss + "\"max_depth\":" + to_string(config.max_depth) + ",";
		ss = ss + "\"learning_rate\":" + to_string(config.learning_rate) + ",";
		ss = ss + "\"min_samples_split\":" + to_string(config.min_samples_split) + ",";
		ss = ss + "\"min_data_in_leaf\":" + to_string(config.min_data_in_leaf) + ",";
		ss = ss + "\"min_child_weight\":" + to_string(config.min_child_weight) + ",";
		ss = ss + "\"colsample_bytree\":" + to_string(config.colsample_bytree) + ",";
		ss = ss + "\"reg_gamma\":" + to_string(config.reg_gamma) + ",";
		ss = ss + "\"reg_lambda\":" + to_string(config.reg_lambda) + ",";
		ss = ss + "\"max_bin\":" + to_string(config.max_bin);
		ss += "}";
		return ss + "}";
	}

	//由json生成Tree
	Tree* XGBoost::LoadModelFromJson(const rapidjson::Value &doc) {
		xgboost::Tree *tree = new Tree();
		if (doc.HasMember("leaf_value")) {
			tree->leaf_value = doc["leaf_value"].GetFloat();
			return tree;
		}
		tree->split_feature = doc["split_feature"].GetInt();
		tree->split_value = doc["split_value"].GetFloat();
		tree->split_gain = doc["split_gain"].GetFloat();
		tree->internal_value = doc["internal_value"].GetFloat();

		if (doc.HasMember("tree_left")) {
			tree->tree_left = LoadModelFromJson(doc["tree_left"]);
		}
		if (doc.HasMember("tree_right")) {
			tree->tree_right = LoadModelFromJson(doc["tree_right"]);
		}
		return tree;
	}
}
