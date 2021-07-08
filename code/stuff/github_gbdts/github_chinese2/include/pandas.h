#pragma once
#include <string>
#include <vector>


namespace pandas {
	struct Dataset {
		std::vector<std::vector<float>> features;
		std::vector<int> labels;
	};

	Dataset ReadCSV(std::string file_path, char sep, float fillna, int n_rows = INT_MAX);
	void SaveCSV(const std::vector<float>& dataset_vect, const std::string file_path);
}