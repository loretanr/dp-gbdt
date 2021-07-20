#include "loss.h"

/* ---------- LSE ---------- */

std::vector<double> LeastSquaresError::compute_gradients(std::vector<double> &y, std::vector<double> &y_pred)
    {
        std::vector<double> gradients(y.size());
        for (size_t i=0; i<y.size(); i++) {
            gradients[i] = y_pred[i] - y[i];
        }
        // limit the numbers of decimals to avoid numeric inconsistencies
        std::transform(gradients.begin(), gradients.end(),
                gradients.begin(), [](double c){ return std::floor(c * 1e15) / 1e15; });
        return gradients;
    }
    
double LeastSquaresError::compute_init_score(std::vector<double> &y)
{
    // for regression simply start with mean
    double sum = std::accumulate(y.begin(), y.end(), 0.0);
    return sum / y.size();
}

/* ---------- Binominal Deviance ---------- */

std::vector<double> BinomialDeviance::compute_gradients(std::vector<double> &y, std::vector<double> &y_pred)
    {
        // positive gradient = expit(y_pred) - y
        // expit(x) (logistic sigmoid function) = 1/(1+exp(-x))
        std::vector<double> gradients(y.size());
        for (size_t i=0; i<y.size(); i++) {
            gradients[i] = 1 / (1 + std::exp(-y_pred[i])) - y[i];
        }
        // limit the numbers of decimals to avoid numeric inconsistencies
        std::transform(gradients.begin(), gradients.end(),
                gradients.begin(), [](double c){ return std::floor(c * 1e15) / 1e15; });
        return gradients;
    }

double BinomialDeviance::compute_init_score(std::vector<double> &y)
{
    std::map<double,double> occurrences;
    for(auto elem : y) {
        try {
            occurrences.at(elem) += 1;
        } catch (const std::out_of_range& oor) {
            // new label encountered, create mapping
            occurrences.insert({elem, 1});
        }
    }
    // just need the smaller value ??? 
    std::set<double, std::greater<double>> occs;
    for(auto &elem : occurrences){
        occs.insert( (double) elem.second / y.size());
    }
    double smaller_value = *occs.rbegin();
    // log(x / (1-x)) is the inverse of the sigmoid (expit) function
    double prediction = std::log(smaller_value / (1- smaller_value));
    return prediction;
}