

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
    T select(bool condition, T a, T b)
    {
        // result = cond * a + !cond * b
        return condition * a + !condition * b;
    }
}


int main() {
    int a = 42;
    int b = 89;

    int *ap = &a;
    int *bp = &b;

    int *c = (int *) constant_time::select(false, (unsigned long) ap, (unsigned long) bp);

    return *c;
}