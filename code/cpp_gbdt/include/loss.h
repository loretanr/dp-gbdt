#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include "utils.h"

class LossFunction
{
public:
    // virtual ~LossFunction() = default;
    virtual std::vector<double> compute_gradients(std::vector<double> &y, std::vector<double> &y_pred){ return {0}; };
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
    // virtual ~LeastSquaresError(){}
};

class BinomialDeviance : public LossFunction
{
public:
    virtual std::vector<double> compute_gradients(std::vector<double> &y, std::vector<double> &y_pred);
    // virtual ~BinomialDeviance(){};
};



#endif /* LOSS_FUNCTION_H */
