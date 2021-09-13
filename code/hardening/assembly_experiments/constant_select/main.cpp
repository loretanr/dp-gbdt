#include <random>

#define TURN_ON_INLINE false

#if TURN_ON_INLINE
    #define USE_INLINE __inline__
#else
    #define USE_INLINE __attribute__((noinline))
#endif

template <typename T>
static USE_INLINE T value_barrier(T a)
{
    volatile T v = a;
    return v;
}

template <typename T>
static USE_INLINE T select(bool condition, T a, T b)
{
    return value_barrier(condition) * value_barrier(a) + value_barrier(!condition) * value_barrier(b);
}


int main()
{
    srand(time(NULL));

    int r2 =  std::rand();
    
    // split_attr = select_int(condition, random_feature, split_attr);
    bool condition = select(condition, 0, r2);

    return condition;
}
