#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include "utils.h"


class LossFunction
{
public:
    virtual std::vector<double> compute_gradients(std::vector<double> &y, std::vector<double> &y_pred){ return {0}; }
    virtual double compute_init_score(std::vector<double> &y, VVD &X){ return 0; }
};


class LeastSquaresError : public LossFunction
{
public:
    virtual std::vector<double> compute_gradients(std::vector<double> &y, std::vector<double> &y_pred)
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
    
    virtual double compute_init_score(std::vector<double> &y, VVD &X)
    {
        // for regression simply start with mean
        double sum = std::accumulate(y.begin(), y.end(), 0.0);
        return sum / y.size();
    }
};


class BinomialDeviance : public LossFunction
{
public:
    virtual std::vector<double> compute_gradients(std::vector<double> &y, std::vector<double> &y_pred)
    {
        // TODO
        std::vector<double> gradients(y.size());
        return gradients;
    }

    virtual double compute_init_score(std::vector<double> &y, VVD &X)
    {
        // TODO
        return 42.69;
    }
};



#endif /* LOSS_FUNCTION_H */
