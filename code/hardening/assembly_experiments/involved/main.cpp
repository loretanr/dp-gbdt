#include <random>

#define TURN_ON_INLINE false

#if TURN_ON_INLINE
    #define USE_INLINE __inline__
#else
    #define USE_INLINE __attribute__((noinline))
#endif

static USE_INLINE int value_barrier(bool a)
{
    volatile int v = a;
    return v;
}

static USE_INLINE int select_int(bool mask, int a, int b)
{
    return (value_barrier(mask) * a) + (value_barrier(!mask) * b);
}
static USE_INLINE int select_int_nobarrier(bool mask, int a, int b)
{
    return ((int) mask * a) + ((int) !mask * b);
}



bool USE_INLINE logic_chain()
{
    bool bool1 = std::rand() % 2;
    bool bool2 = std::rand() % 2;

    bool condition = bool1 | bool2;  // problem
    return condition;
}


int main()
{

    srand(time(NULL));

    bool condition = logic_chain();

    int random_feature = 89;
    int split_attr = 42;
    
    // split_attr = select_int(condition, random_feature, split_attr);
    split_attr = select_int_nobarrier(condition, random_feature, split_attr);

    return split_attr;
}
