#include <iostream>

#include "DifferentiallyPrivateTree.h"

using namespace std;

int main()
{
    ModelParams parammmms;
    parammmms.delta_g = 0.42;
    parammmms.use_bfs = true;
    parammmms.max_depth = 600000;

    DifferentiallyPrivateTree dpt = DifferentiallyPrivateTree(parammmms);
    cout << "hello MA world" << endl;
}