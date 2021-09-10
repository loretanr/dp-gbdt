#ifndef CONSTANT_TIME_H
#define CONSTANT_TIME_H


// note -O0 turns off inlining anyways!
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
    static USE_INLINE T value_barrier(T a)
    {
        // volatile -> hint to the compiler that "a" might be changed 
        //from somewhere outside. -> optimization disincentive
        volatile T v = a;
        return v;
    }

    template <typename T>
    static USE_INLINE T select(bool condition, T a, T b)
    {
        return value_barrier(condition) * value_barrier(a) + value_barrier(!condition) * value_barrier(b);
    }

    // static USE_INLINE double select_double(bool mask, double a, double b)
    // {
    //     return (value_barrier(mask) * a) + (value_barrier(!mask) * b);
    // }


    /** Logical operators */

    static USE_INLINE bool logical_or(bool a, bool b)
    {
        // use bitwise for const time
        return (value_barrier(a) | value_barrier(b));
    }

}

#endif /* CONSTANT_TIME_H */
