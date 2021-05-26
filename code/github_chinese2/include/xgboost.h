#pragma once
#include <vector>
#include "decision_tree.h"
#include "tree.h"
#include "config.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
using namespace rapidjson;


namespace xgboost {
	struct Gradients {
		float grad;
		float hess;
	};

	class XGBoost {
	public:
		XGBoost(Config conf);
		~XGBoost();
		std::vector<Tree*> trees;
		void fit(const std::vector<std::vector<float>>& features, const std::vector<int>& labels);
		std::vector<float> PredictProba(const std::vector<float>& features);
		const Config config;
		float pred_0;
		std::string SaveModelToString();
		Tree* LoadModelFromJson(const rapidjson::Value &doc);

	private:
		std::vector<float> grad;
		std::vector<float> hess;
		Gradients CalculateGradHess(int y, float y_pred);
	};
}
