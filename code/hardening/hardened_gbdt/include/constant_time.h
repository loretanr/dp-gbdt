#ifndef CONSTANT_TIME_H
#define CONSTANT_TIME_H

namespace constant_time
{
    /*
    * hint to the compiler that "a" might be changed from somewhere
    * outside. -> optimization disincentive
    */
    static __inline__ bool value_barrier(bool a)
    {
        volatile int v = a;
        return v;
    }

    static __inline__ bool select(bool mask, int a, int b)
    {
        return (bool) (value_barrier(mask) & a) + (value_barrier(!mask) & b);
    }



    // static __inline__ unsigned int value_barrier(unsigned int a)
    // {
    //     volatile unsigned int v = a;
    //     return v;
    // }

    // static __inline__ unsigned int select(unsigned int mask, unsigned int a, unsigned int b)
    // {
    //     return (value_barrier(mask) & a) | (value_barrier(~mask) & b);
    // }

    // static __inline__ int select_int(unsigned int mask, int a, int b)
    // {
    //     return (int) select(mask, (unsigned)(a), (unsigned)(b));
    // }
}

#endif /* CONSTANT_TIME_H */
