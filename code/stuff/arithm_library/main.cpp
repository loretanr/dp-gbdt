#include <iostream>
#include "constant_time.h"


int main()
{
    double a = 1, b = 2, c;

    c = a + b;

    std::cout << c << std::endl;

    c = a * b;

    std::cout << c << std::endl;

    c = constant_time::plus(a, b);

    std::cout << c << std::endl;

    return c;

    // TODO do the floatmath import here
}