#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include "utils.h"
using namespace std;


//计算模型AUC
float CalculateAUC(vector<int>& labels, vector<float>& pvalues) {
	int n_bins = labels.size() <= 10000 ? 1000 : 100;

	int count_positive = accumulate(labels.begin(), labels.end(), 0);
	int count_negative = labels.size() - count_positive;
	int total_case = count_positive * count_negative;
	vector<int> pos_histogram(n_bins, 0);
	vector<int> neg_histogram(n_bins, 0);

	float bin_width = 1.0 / n_bins;
	for (size_t i = 0; i < labels.size(); ++i) {
		int ix = int(pvalues[i] / bin_width);
		if (labels[i] == 1) {
			pos_histogram[ix] += 1;
		}
		else {
			neg_histogram[ix] += 1;
		}
	}
	int accumulated_neg = 0;
	float satisfied_pair = 0;
	for (int i = 0; i < n_bins; ++i) {
		satisfied_pair += (pos_histogram[i] * accumulated_neg + pos_histogram[i] * neg_histogram[i] * 0.5);
		accumulated_neg += neg_histogram[i];
	}
	return satisfied_pair / float(total_case);
}

//计算模型KS
float CalculateKS(vector<int>& labels, vector<float>& pvalues) {
	vector<float> values_unique = pvalues;
	sort(values_unique.begin(), values_unique.end());
	values_unique.erase(unique(values_unique.begin(), values_unique.end()), values_unique.end());

	float tpr, fpr, distance;
	float max_distance = 0;
	for (float cut_point : values_unique) {
		float tp = 0.0001;
		float fn = 0.0001;
		float fp = 0.0001;
		float tn = 0.0001;

		for (size_t i = 0; i < labels.size(); ++i) {
			if ((pvalues[i] >= cut_point) && (labels[i] == 1)) {
				tp += 1;
			}
			else if ((pvalues[i] >= cut_point) && (labels[i] != 1)) {
				fp += 1;
			}
			else if ((pvalues[i] < cut_point) && (labels[i] == 1)) {
				fn += 1;
			}
			else {
				tn += 1;
			}
		}
		tpr = tp / (tp + fn);
		fpr = fp / (fp + tn);
		distance = tpr - fpr;
		if (distance > max_distance) {
			max_distance = distance;
		}
	}
	return max_distance;
}

//计算模型准确率ACC
float CalculateACC(vector<int>& labels, vector<float>& pvalues) {
	int count_right = 0;
	for (size_t i = 0; i < labels.size(); ++i) {
		if ((labels[i] == 0 && pvalues[i] < 0.5) || (labels[i] == 1 && pvalues[i] >= 0.5)) {
			count_right += 1;
		}
	}
	return (float)count_right / labels.size();
}

