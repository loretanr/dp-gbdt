// #include <memory>
// #include <map>
// #include <numeric>
// #include <sstream>
// #include <fstream>
// #include <string>
// #include <algorithm>
// #include "parser.h"


// /* Parsing:
//     - the data file needs to be comma separated
//     - the dataset must not contain missing values!
//     - you need to specify stuff like size, which features are numerical/categorical,
//       which feature is the target and which features you want to drop.

//     Given these requirements, it should be easy to add new datasets in the same 
//     fashion as the ones below. But make sure to double check what you get.
// */


// sgx_dataset sgx_Parser::get_abalone(sgx_modelparams &parameters, size_t num_samples)
// {
//     std::string file = "datasets/real/abalone.data";
//     std::string name = "abalone";
//     int num_rows = 4177;
//     int num_cols = 9;
//     // std::shared_ptr<Regression> task(new Regression());
//     char *task = "regression";
//     int *num_idx = {1,2,3,4,5,6,7};
//     int *cat_idx = {0};
//     int target_idx = 8;
//     int * drop_idx = {};

//     return parse_file(file, name, num_rows, num_cols, num_samples, task, num_idx,
//         cat_idx, target_idx, drop_idx, parameters, use_default_params);
// }


// // DataSet Parser::get_YearPredictionMSD(std::vector<ModelParams> &parameters, 
// //         size_t num_samples, bool use_default_params)
// // {
// //     std::string file = "datasets/real/YearPredictionMSD.txt";
// //     std::string name = "yearMSD";
// //     int num_rows = 515345;
// //     int num_cols = 91;
// //     std::shared_ptr<Regression> task(new Regression());
// //     std::vector<int> num_idx(90);
// //     std::iota(std::begin(num_idx)++, std::end(num_idx), 1); // num_idx = {1,...,90}
// //     std::vector<int> cat_idx = {};
// //     std::vector<int> target_idx = {0};
// //     std::vector<int> drop_idx = {};

// //     return parse_file(file, name, num_rows, num_cols, num_samples, task, num_idx,
// //         cat_idx, target_idx, drop_idx, parameters, use_default_params);
// // }


// // DataSet Parser::get_adult(std::vector<ModelParams> &parameters,
// //         size_t num_samples, bool use_default_params)
// // {
// //     std::string file = "datasets/real/adult.data";
// //     std::string name = "adult";
// //     int num_rows = 48842;
// //     int num_cols = 90;
// //     std::shared_ptr<BinaryClassification> task(new BinaryClassification());
// //     std::vector<int> num_idx = {0,2,4,10,11,12};
// //     std::vector<int> cat_idx = {1,3,5,6,7,8,9,13};
// //     std::vector<int> target_idx = {14};
// //     std::vector<int> drop_idx = {};

// //     return parse_file(file, name, num_rows, num_cols, num_samples, task, num_idx,
// //         cat_idx, target_idx, drop_idx, parameters, use_default_params);
// // }



// /** Utility functions */

// std::vector<std::string> sgx_Parser::split_string(const std::string &s, char delim)
// {
//     std::vector<std::string> result;
//     std::stringstream ss(s);
//     std::string item;
//     while (std::getline(ss, item, delim)) {
//         result.push_back(item);
//     }
//     return result;
// }


// sgx_dataset sgx_Parser::parse_file(std::string dataset_file, std::string dataset_name, int num_rows,
//         int num_cols, int num_samples, char *task, int *num_idx,
//         int *cat_idx, int target_idx, int *drop_idx,
//         sgx_modelparams &modelparams)
// {
//     std::ifstream infile(dataset_file);
//     std::string line;
//     num_samples = std::min(num_samples, num_rows);

//     modelparams.num_idx = num_idx;
//     modelparams.cat_idx = cat_idx;
//     modelparams.task = task;

//     int num_cols = num_idx.size() + cat_idx.size() - drop_idx.size();
//     double *X = (double *) malloc(num_cols * num_samples * sizeof(double));
//     double *y = (double *) malloc(num_samples * sizeof(double));

//     // parse dataset, label-encode categorical features
//     int current_index = 0;
//     std::vector<std::map<std::string,float>> mappings(num_cols + 1); // last (additional) one is for y

//     while (std::getline(infile, line,'\n') && current_index < num_samples) {
//         std::stringstream ss(line);
//         std::vector<std::string> strings = split_string(line, ',');
//         // double *X_row = (double *) malloc(num_cols * sizeof(double));

//         // go through each column
//         for(size_t i=0; i<strings.size(); i++){

//             // is it a drop column?
//             if (std::find(drop_idx.begin(), drop_idx.end(), i) != drop_idx.end()) {
//                 continue;
//             }

//             // y
//             if(std::find(target_idx.begin(), target_idx.end(), i) != target_idx.end()){
//                 if (dynamic_cast<Regression*>(task.get())) {
//                     // regression -> y is numerical
//                     y[current_index] = stof(strings[i]);
//                 } else {
//                     try { // categorical
//                         float dummy_value = mappings.back().at(strings[i]);
//                         y[current_index] = dummy_value;
//                     } catch (const std::out_of_range& oor) {
//                         // new label encountered, create mapping
//                         mappings.back().insert({strings[i], mappings.back().size()});
//                         float dummy_value = mappings.back().at(strings[i]);
//                         y[current_index] = dummy_value;
//                     }
//                 }
//                 continue;
//             }

//             int current_col = 0;

//             // X
//             if (std::find(num_idx.begin(), num_idx.end(), i) != num_idx.end()) {
//                 // numerical feature
//                 X[current_index * num_cols + current_col++] = stof(strings[i]);
//                 // X_row.push_back(stof(strings[i]));
//             } else {
//                 // categorical feature, do label-encoding
//                 try {
//                     float dummy_value = mappings[i].at(strings[i]);
//                     // X_row.push_back(dummy_value);
//                     X[current_index * num_cols + current_col++] = dummy_value;
//                 } catch (const std::out_of_range& oor) {
//                     // new label encountered, create mapping
//                     mappings[i].insert({strings[i], mappings[i].size()});
//                     float dummy_value = mappings[i].at(strings[i]);
//                     // X_row.push_back(dummy_value);
//                     X[current_index * num_cols + current_col++] = dummy_value;
//                 }
//             }
//         }
//         // X.push_back(X_row);
//         current_index++;
//     }

//     sgx_dataset dataset;
//     dataset.X = X;
//     dataset.y = y;
//     dataset.name = std::string(dataset_name) + std::string("_size_") + std::to_string(num_samples);
//     return dataset;
// }
