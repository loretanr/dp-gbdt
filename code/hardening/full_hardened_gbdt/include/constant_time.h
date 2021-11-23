#ifndef CONSTANT_TIME_H
#define CONSTANT_TIME_H

#include <vector>

// extern "C" {
//     #include "ftfp.h"
// }


// note, -O0 turns off inlining anyways!
#define DISABLE_INLINE true

#if not DISABLE_INLINE
    #define USE_INLINE __inline__
#else
    #define USE_INLINE __attribute__((noinline))
#endif


namespace constant_time
{
    // probably unnecessary, but keeping it to be safe
    template <typename T>
    T value_barrier(T a)
    {
        // volatile -> hint to the compiler that "a" might be changed 
        //from somewhere outside. -> optimization disincentive
        volatile T v = a;
        return v;
    }

    /** oblivious assign aka select */

    template <typename T>
    USE_INLINE T select(bool condition, T a, T b)
    {
        // result = cond * a + !cond * b
        return value_barrier(condition) * value_barrier(a) + value_barrier(!condition) * value_barrier(b);
    }


    /** Logical operators */

    USE_INLINE bool logical_or(bool a, bool b);

    USE_INLINE bool logical_and(bool a, bool b);

    USE_INLINE bool logical_not(bool a);


    /** constant time max/min */

    template <typename T>
    USE_INLINE T max(T a, T b)
    {
        return select(value_barrier(a) >= value_barrier(b), value_barrier(a) , value_barrier(b));
    }

    template <typename T>
    USE_INLINE T min(T a, T b)
    {
        return select(value_barrier(a) <= value_barrier(b), value_barrier(a) , value_barrier(b));
    }


    /** constant time sort */

    template <typename T>
    USE_INLINE void sort(std::vector<T> &vec)
    {
        // O(n^2) bubblesort, because it's easy to harden and it's not a performance critical task
        for (size_t n=vec.size(); n>1; --n){
            for (size_t i=0; i<n-1; ++i){
                // swap pair if necessary
                bool condition = vec[i] > vec[i+1];
                T temp = vec[i];
                vec[i] = constant_time::select(condition, vec[i+1], vec[i]);
                vec[i+1] = constant_time::select(condition, temp, vec[i+1]);
            }
        }
    }


    // bool smaller(double a, double b) {
    //     fixed aa = fix_convert_from_double(a);
    //     fixed bb = fix_convert_from_double(b);
    //     return (bool) fix_lt(a, b);
    // }

}

#endif /* CONSTANT_TIME_H */
