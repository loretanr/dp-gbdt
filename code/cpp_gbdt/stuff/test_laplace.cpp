#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

int main()
{
    // DISTRIBUTION test
    double scale = 4;
    vector<double> samples;
    srand(time(NULL));
    Laplace lap(scale, rand());
    for(int i=0; i<10000; i++){
        samples.push_back(lap.return_a_random_variable(scale));
    }
    ofstream myfile;
    myfile.open ("laplace.txt");
    for(auto sample : samples) {
        myfile << sample << " ";
    }
    myfile.close();
    exit(0);
}