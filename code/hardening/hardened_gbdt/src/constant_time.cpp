#include "constant_time.h"





USE_INLINE bool constant_time::logical_or(bool a, bool b)
{
    // use bitwise for const time
    return (value_barrier_v2(a) | value_barrier_v2(b));
}

USE_INLINE bool constant_time::logical_and(bool a, bool b)
{
    // use bitwise for const time
    return (value_barrier(a) & value_barrier(b));
}

USE_INLINE bool constant_time::logical_not(bool a)
{
    // use bitwise for const time
    return (bool) (value_barrier((unsigned) a) ^ 1u);
}

