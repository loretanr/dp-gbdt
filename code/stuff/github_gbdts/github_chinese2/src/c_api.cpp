#include <iostream>
#include <vector>
#include <time.h>
#include "c_api.h"
#include "pandas.h"
#include "xgboost.h"
#include "pandas.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
using namespace std;
using namespace pandas;
using namespace xgboost;
using namespace rapidjson;


#define API_BEGIN() try {
#define API_END() } catch(std::runtime_error &_except_) { return -1; } return 0;

//训练模型接口
XGB_DLL int BoosterTrain(Config *conf, const float *data, const int *label, int nrow, int ncol, Booster *booster) {
	API_BEGIN();
	vector< vector<float> > features(nrow, vector<float>(ncol));
	vector<int> labels;

	for (int i = 0; i < ncol; ++i, data += nrow) {
		for (int j = 0; j < nrow; ++j) {
			features[j][i] = data[j];
		}
	}
	for (int k = 0; k < nrow; ++k)
		labels.push_back(label[k]);
	
	XGBoost *model = new XGBoost(*conf);
	model->fit(features, labels);
	*booster = model;
	API_END();
}

//模型预测接口
XGB_DLL int BoosterPredict(const float *data, int nrow, int ncol, Booster *booster, float *result) {
	API_BEGIN();
	vector< vector<float> > features(nrow, vector<float>(ncol));

	for (int i = 0; i < ncol; ++i, data += nrow) {
		for (int j = 0; j < nrow; ++j) {
			features[j][i] = data[j];
		}
	}

	XGBoost *model = static_cast<XGBoost*>(*booster);
	for (size_t i = 0; i < features.size(); ++i)
		result[i] = model->PredictProba(features[i])[1];
	API_END();
}


//模型打包为string
XGB_DLL int BoosterSaveModelToString(Booster *booster, char *out_str) {
	API_BEGIN();
	XGBoost *model = reinterpret_cast<XGBoost*>(*booster);
	std::string out_model = model->SaveModelToString();
	std::memcpy(out_str, out_model.c_str(), static_cast<int64_t>(out_model.size()) + 1);
	API_END();
}

//string解析成模型
XGB_DLL int BoosterLoadModelFromString(char *input_str, Booster *booster) {
	API_BEGIN();
	std::string jsonText = input_str;
	rapidjson::Document doc;
	doc.Parse(jsonText.c_str());

	rapidjson::Value &param = doc["Param"];
	Config config;
	config.n_estimators = param["n_estimators"].GetInt();
	config.learning_rate = param["learning_rate"].GetFloat();
	config.max_depth = param["max_depth"].GetInt();
	config.min_samples_split = param["min_samples_split"].GetInt();
	config.min_data_in_leaf = param["min_data_in_leaf"].GetInt();
	config.reg_gamma = param["reg_gamma"].GetFloat();
	config.reg_lambda = param["reg_lambda"].GetFloat();
	config.colsample_bytree = param["colsample_bytree"].GetFloat();
	config.min_child_weight = param["min_child_weight"].GetFloat();
	config.max_bin = param["max_bin"].GetInt();

	XGBoost *model = new XGBoost(config);
	rapidjson::Value &tree_list = doc["Trees"];
	if (tree_list.IsArray()) {
		for (size_t i = 0; i < tree_list.Size(); ++i) {
			model->trees.push_back(model->LoadModelFromJson(tree_list[i]));
		}
	}
	*booster = model;
	API_END();
}