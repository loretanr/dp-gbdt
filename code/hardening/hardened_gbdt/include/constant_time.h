#ifndef CONSTANT_TIME_H
#define CONSTANT_TIME_H


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
    USE_INLINE T value_barrier(T a)
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


    /** constant time max / min */

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

}

#endif /* CONSTANT_TIME_H */
