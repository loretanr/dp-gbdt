#include "loss.h"
#include <set>
#include <map>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

extern bool VERIFICATION_MODE;

/* ---------- Regression ---------- */

double Regression::compute_init_score(std::vector<double> &y)
{
    // mean
    double sum = std::accumulate(y.begin(), y.end(), 0.0);
    return sum / y.size();
}

std::vector<double> Regression::compute_gradients(std::vector<double> &y, std::vector<double> &y_pred)
    {
        // TODO negative or positive gradient? what is this?
        std::vector<double> gradients(y.size());
        for (size_t i=0; i<y.size(); i++) {
            gradients[i] = y_pred[i] - y[i];
        }
        
        if(VERIFICATION_MODE){
            // limit the numbers of decimals to avoid numeric inconsistencies
            std::transform(gradients.begin(), gradients.end(),
                    gradients.begin(), [](double c){ return std::floor(c * 1e15) / 1e15; });
        }

        return gradients;
    }
    
double Regression::compute_score(std::vector<double> &y, std::vector<double> &y_pred)
{
    // RMSE
    std::transform(y.begin(), y.end(), 
            y_pred.begin(), y_pred.begin(), std::minus<double>());
    std::transform(y_pred.begin(), y_pred.end(),
            y_pred.begin(), [](double &c){return std::pow(c,2);});
    double average = std::accumulate(y_pred.begin(),y_pred.end(), 0.0) / y_pred.size();
    double rmse = std::sqrt(average);
    return rmse;
}


/* ---------- Binary Classification ---------- */

// Uses Binomial Deviance
// TODO, link between theory (expit/logit) and code

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
    // just need the smaller value   ????? TODO why
    std::set<double, std::greater<double>> occs;
    for(auto &elem : occurrences){
        occs.insert( (double) elem.second / y.size());
    }
    double smaller_value = *occs.rbegin();
    // "log(x / (1-x)) is the inverse of the sigmoid (expit) function"
    double prediction = std::log(smaller_value / (1- smaller_value));
    return prediction;
}

std::vector<double> BinaryClassification::compute_gradients(std::vector<double> &y, std::vector<double> &y_pred)
    {
        // positive gradient: expit(y_pred) - y
        // expit(x): (logistic sigmoid function) = 1/(1+exp(-x))
        std::vector<double> gradients(y.size());
        for (size_t i=0; i<y.size(); i++) {
            gradients[i] = 1 / (1 + std::exp(-y_pred[i])) - y[i];
        }

        if(VERIFICATION_MODE){
            // limit the numbers of decimals to avoid numeric inconsistencies
            std::transform(gradients.begin(), gradients.end(),
                    gradients.begin(), [](double c){ return std::floor(c * 1e15) / 1e15; });
        }
        return gradients;
    }

double BinaryClassification::compute_score(std::vector<double> &y, std::vector<double> &y_pred)
{
    // classification task -> transform continuous predictions back to labels
    std::transform(y_pred.begin(), y_pred.end(), // expit
        y_pred.begin(), [](double &c){ return 1 / (1 + std::exp(-c)); });
    for(auto &elem : y_pred){
        elem = (elem < 1.-elem) ? 0.0 : 1.0;
    }

    // std::cout << "1-preds: " << std::count(y_pred.begin(), y_pred.end(), 1.0) << std::endl;

    // accuracy
    std::vector<bool> correct_preds(y.size());
    for(size_t i=0; i<y.size();i++) {
        correct_preds[i] = (y[i] == y_pred[i]);
    }
    double true_preds = std::count(correct_preds.begin(), correct_preds.end(), true);
    return true_preds / y.size();
}