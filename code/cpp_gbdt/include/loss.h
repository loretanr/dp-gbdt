#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include <map>
#include <set>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

// abstract class
class LossFunction
{
public:
    virtual std::vector<double> compute_gradients(std::vector<double> &y, std::vector<double> &y_pred) = 0;
    virtual double compute_init_score(std::vector<double> &y) = 0;
};


class LeastSquaresError : public LossFunction
{
public:
    virtual std::vector<double> compute_gradients(std::vector<double> &y, std::vector<double> &y_pred);
    virtual double compute_init_score(std::vector<double> &y);
};


class BinomialDeviance : public LossFunction
{
public:
    virtual std::vector<double> compute_gradients(std::vector<double> &y, std::vector<double> &y_pred);
    virtual double compute_init_score(std::vector<double> &y);
};


#endif /* LOSS_FUNCTION_H */
