#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include <vector>

// abstract class
class Task
{
public:
    virtual std::vector<double> compute_gradients(std::vector<double> &y, std::vector<double> &y_pred) = 0;
    virtual double compute_init_score(std::vector<double> &y) = 0;
    virtual double compute_score(std::vector<double> &y, std::vector<double> &y_pred) = 0;
};


// uses LSE as cost/loss function
class Regression : public Task
{
public:

    // TODO name?
    virtual std::vector<double> compute_gradients(std::vector<double> &y, std::vector<double> &y_pred);
    
    // mean
    virtual double compute_init_score(std::vector<double> &y);

    // RMSE
    virtual double compute_score(std::vector<double> &y, std::vector<double> &y_pred);
};

// uses Binomial Deviance as cost/loss function
class BinaryClassification : public Task
{
public:

    // expit
    virtual std::vector<double> compute_gradients(std::vector<double> &y, std::vector<double> &y_pred);
    
    // logit
    virtual double compute_init_score(std::vector<double> &y);

    // accuracy
    virtual double compute_score(std::vector<double> &y, std::vector<double> &y_pred);
};


#endif /* LOSS_FUNCTION_H */