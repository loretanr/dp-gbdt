#pragma once
#include <vector>
#include <iostream>


namespace numpy {
	//求分位数，仿照numpy.Percentile
	template<typename T, typename VECT_T = std::vector<T>>
	float Percentile(const VECT_T& vect, T p) {
		if (!p) {
			return (float)vect[0];
		}
		else if (100 - p < 1e-5) {
			return (float)vect[vect.size() - 1];
		}
		else {
			float temp = (vect.size() - 1) * p / 100.0 + 1;
			int pos_integer = floor(temp);
			float pos_decimal = temp - pos_integer;
			float res = vect[pos_integer - 1] + (vect[pos_integer] - vect[pos_integer - 1]) * pos_decimal;
			return (float)res;
		}
	}

	//求等差数列，仿照numpy.Linspace
	template<typename T>
	std::vector<float> Linspace(T start, T end, int n) {
		float step = (end - start) * 1.0 / (n - 1);
		std::vector<float> res;
		for (int i = 0; i < n; ++i) {
			res.push_back(start + i*step);
		}
		return res;
	}
}