#include <random>
#include <string>

int main()
{
    srand(time(NULL));

    int i2 = std::rand() % 42;

    bool result = not i2;
    
    return result;
}