#pragma once
#include <vector>
#define XGB_EXTERN_C extern "C"
#define XGB_DLL XGB_EXTERN_C __declspec(dllexport)
#include "config.h"
#include "pandas.h"
#include "xgboost.h"


typedef xgboost::XGBoost *Booster;
XGB_DLL int BoosterTrain(xgboost::Config *config, const float *feature, const int *label, int nrow, int ncol, Booster *booster);
XGB_DLL int BoosterPredict(const float *feature, int nrow, int ncol, Booster *booster, float *result);
XGB_DLL int BoosterSaveModelToString(Booster *booster, char *out_str);
XGB_DLL int BoosterLoadModelFromString(char *input_str, Booster *booster);
