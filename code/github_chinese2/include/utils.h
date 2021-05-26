#pragma once
#include <vector>


float CalculateAUC(std::vector<int>& labels, std::vector<float>& pvalues);
float CalculateKS(std::vector<int>& labels, std::vector<float>& pvalues);
float CalculateACC(std::vector<int>& labels, std::vector<float>& pvalues);

