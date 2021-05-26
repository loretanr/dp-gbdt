#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include "pandas.h"
using namespace std;


namespace pandas {
	//读取csv文件，仿照pandas.ReadCSV
	Dataset ReadCSV(std::string file_path, char sep, float fillna, int n_rows) {
		Dataset dataset;
		vector<vector<float>> features;
		vector<int> labels;

		ifstream ifs(file_path);
		std::string line;
		int count_rows = 0;
		while (getline(ifs, line) && (count_rows < n_rows)) {
			++count_rows;

			if (!line.empty()) {
				stringstream ss(line);
				vector<float> vect_line;
				std::string tmp;
				while (getline(ss, tmp, sep)) {
					if (tmp == "") {
						vect_line.push_back(fillna);
					}
					else {
						vect_line.push_back(stof(tmp));
					}
				}
				labels.push_back(int(vect_line.back()));
				vect_line.pop_back();
				features.push_back(vect_line);
			}
		}

		dataset = { features, labels };
		return dataset;
	}

	// 将给定的向量写到csv文件
	void SaveCSV(const vector<float>& dataset_vect, const std::string file_path) {
		ofstream outFile;
		outFile.open(file_path, ios::out);
		for (float value : dataset_vect) {
			outFile << value << endl;
		}
		outFile.close();
	}
}
