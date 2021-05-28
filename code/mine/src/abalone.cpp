#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "DPTree.h"
#include "utils.h"

using namespace std;

int main()
{
    // Parse dataset
    ifstream infile("data/abalone.data");
    string line;
    float val;
    vector<vector<float>> X; // TODO think about row/columnwise what makes sense
    vector<float> y;

    int lineIdx = 0;
    while (getline(infile, line,'\n')) {
        stringstream ss(line);
        int colIdx = 0;
        vector<float> X_row;
        while(ss >> val) {
            
            if(colIdx < 8){
                X_row.push_back(val);
            } else {
                y.push_back(val);
            }
            colIdx++;
        }
        X.push_back(X_row);
        lineIdx++;
    }
    DataSet dataset = DataSet(&X, &y);

    ModelParams parammmms;
    parammmms.delta_g = 0.42;
    parammmms.use_bfs = true;
    parammmms.max_depth = 600000;

    DPTree dpt = DPTree(&parammmms, &dataset);
    cout << "hello MA world" << endl;
}