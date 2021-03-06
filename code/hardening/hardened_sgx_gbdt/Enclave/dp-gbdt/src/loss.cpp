#include "loss.h"
#include <set>
#include <map>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "utils.h"
#include "Enclave.h" // sgx printf


/* ---------- Regression ---------- */

std::vector<double> Regression::compute_gradients(std::vector<double> &y, std::vector<double> &y_pred)
    {
        std::vector<double> gradients(y.size());
        for (size_t i=0; i<y.size(); i++) {
            gradients[i] = y_pred[i] - y[i];
        }
        return gradients;
    }
    
double Regression::compute_init_score(std::vector<double> &y)
{
    // mean
    double sum = std::accumulate(y.begin(), y.end(), 0.0);
    return sum / (double) y.size();
}

double Regression::compute_score(std::vector<double> &y, std::vector<double> &y_pred)
{
    // sgx_printf("y: ");
    // for(auto elem : y){
    //     sgx_printf("%.2f ", elem);
    // } sgx_printf("\n");
    // sgx_printf("y_pred: ");
    // for(auto elem : y_pred){
    //     sgx_printf("%.2f ", elem);
    // } sgx_printf("\n");

    // RMSE
    std::transform(y.begin(), y.end(), 
            y_pred.begin(), y_pred.begin(), std::minus<double>());
    std::transform(y_pred.begin(), y_pred.end(),
            y_pred.begin(), [](double &c){return std::pow(c,2);});
    double average = std::accumulate(y_pred.begin(),y_pred.end(), 0.0) / (double) y_pred.size();
    double rmse = std::sqrt(average);
    return rmse;
}


/* ---------- Binary Classification ---------- */

std::vector<double> BinaryClassification::compute_gradients(std::vector<double> &y, std::vector<double> &y_pred)
    {
        // positive gradient: expit(y_pred) - y
        // expit(x): (logistic sigmoid function) = 1/(1+exp(-x))
        std::vector<double> gradients(y.size());
        for (size_t i=0; i<y.size(); i++) {
            gradients[i] = 1 / (1 + std::exp(-y_pred[i])) - y[i];
        }

        return gradients;
    }

double BinaryClassification::compute_init_score(std::vector<double> &y)
{
    // count how many samples are in each of the 2 classes
    std::map<double,double> occurrences;
    for(auto elem : y) {
        try {
            occurrences.at(elem) += 1;
        } catch (const std::out_of_range& oor) {
            // new label encountered, create mapping
            occurrences.insert({elem, 1});
        }
    }
    // just need the smaller value
    std::set<double, std::greater<double>> occs;
    for(auto &elem : occurrences){
        occs.insert( (double) elem.second / (double) y.size());
    }
    double smaller_value = *occs.rbegin();
    // "log(x / (1-x)) is the inverse of the sigmoid (expit) function"
    double prediction = std::log(smaller_value / (1- smaller_value));
    return prediction;
}

double BinaryClassification::compute_score(std::vector<double> &y, std::vector<double> &y_pred)
{
    // sgx_printf("y: ");
    // for(auto elem : y){
    //     sgx_printf("%.2f ", elem);
    // } sgx_printf("\n");
    // sgx_printf("y_pred: ");
    // for(auto elem : y_pred){
    //     sgx_printf("%.2f ", elem);
    // } sgx_printf("\n");

    // classification task -> transform continuous predictions back to labels
    std::transform(y_pred.begin(), y_pred.end(), // expit
        y_pred.begin(), [](double &c){ return 1 / (1 + std::exp(-c)); });
    for(auto &elem : y_pred){
        elem = (elem < 1-elem) ? 0 : 1;
    }

    // accuracy
    std::vector<bool> correct_preds(y.size());
    for(size_t i=0; i<y.size();i++) {
        correct_preds[i] = (double_equality(y[i], y_pred[i]));
    }
    double true_preds = (double) std::count(correct_preds.begin(), correct_preds.end(), true);
    return true_preds / (double) y.size();
}