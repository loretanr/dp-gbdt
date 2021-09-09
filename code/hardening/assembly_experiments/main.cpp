#include <random>

bool __attribute__((noinline)) logic_chain()
{
    bool bool1 = std::rand() % 2;
    bool bool2 = std::rand() % 2;

    bool condition = bool1 or bool2;  // problem
    return condition;
}


int main()
{

    srand(time(NULL));

    bool condition = logic_chain();

    int random_feature = 5;
    int split_attr = 42;
    
    // no problem under -O0 at least
    split_attr = condition * random_feature + !condition * split_attr;

    return split_attr;
}
