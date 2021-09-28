#include "constant_time.h"

extern "C" {
    #include "libfixedtime/ftfp.h"
}

double constant_time::plus(double a, double b)
{
    fixed res = fix_add(fix_convert_from_double(a),  fix_convert_from_double(b));
    return  fix_convert_to_double(res);
}