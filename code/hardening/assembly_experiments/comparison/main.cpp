#include <random>
#include <string>

int main()
{
    srand(time(NULL));
    std::string s1 = std::to_string( std::rand() );
    std::string s2 = std::to_string( std::rand() );

    bool result = s1 == s2;
    
    return result;
}