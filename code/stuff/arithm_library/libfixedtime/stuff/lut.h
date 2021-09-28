#ifndef LUT_H
#define LUT_H

#include "base.h"
#include "internal.h"

#define INT_INV_LUT fix_internal LUT_int_inv_integer[25] = { \
  0x1000000000000000,\
  0x1000000000000000,\
  0x0800000000000000,\
  0x0555555555555555,\
  0x0400000000000000,\
  0x0333333333333333,\
  0x02aaaaaaaaaaaaab,\
  0x0249249249249249,\
  0x0200000000000000,\
  0x01c71c71c71c71c7,\
  0x019999999999999a,\
  0x01745d1745d1745d,\
  0x0155555555555555,\
  0x013b13b13b13b13b,\
  0x0124924924924925,\
  0x0111111111111111,\
  0x0100000000000000,\
  0x00f0f0f0f0f0f0f1,\
  0x00e38e38e38e38e4,\
  0x00d79435e50d7943,\
  0x00cccccccccccccd,\
  0x00c30c30c30c30c3,\
  0x00ba2e8ba2e8ba2f,\
  0x00b21642c8590b21,\
  0x00aaaaaaaaaaaaab\
};

#define FIX_LN_COEF_0 ((fix_internal) 0xfffffffffffffee4)
#define FIX_LN_COEF_1 ((fix_internal) 0x0fffffffffffc0f3)
#define FIX_LN_COEF_2 ((fix_internal) 0xf8000000000205ea)
#define FIX_LN_COEF_3 ((fix_internal) 0x0555555555ba7707)
#define FIX_LN_COEF_4 ((fix_internal) 0xfbfffffffe5abcce)
#define FIX_LN_COEF_5 ((fix_internal) 0x0333333301c8c3bb)
#define FIX_LN_COEF_6 ((fix_internal) 0xfd555555e268cd1b)
#define FIX_LN_COEF_7 ((fix_internal) 0x0249249d74e41b40)
#define FIX_LN_COEF_8 ((fix_internal) 0xfdffffe742f46443)
#define FIX_LN_COEF_9 ((fix_internal) 0x01c71b097a1e64d0)
#define FIX_LN_COEF_10 ((fix_internal) 0xfe6668ff80d69106)
#define FIX_LN_COEF_11 ((fix_internal) 0x017479001dc22308)
#define FIX_LN_COEF_12 ((fix_internal) 0xfeaa7db83b4a399a)
#define FIX_LN_COEF_13 ((fix_internal) 0x0139aca1896e02c0)
#define FIX_LN_COEF_14 ((fix_internal) 0xfedd75ab312e9d93)
#define FIX_LN_COEF_15 ((fix_internal) 0x011d13cd3e569e29)
#define FIX_LN_COEF_16 ((fix_internal) 0xfef008c734fea1d7)
#define FIX_LN_COEF_17 ((fix_internal) 0x00ac4f012156db32)
#define FIX_LN_COEF_18 ((fix_internal) 0xff716896527419a6)
#define FIX_LN_COEF_19 ((fix_internal) 0x01d722424977647a)
#define FIX_LN_COEF_20 ((fix_internal) 0xfe0997fdbdf61e1d)
#define FIX_LN_COEF_21 ((fix_internal) 0xfe7f51e227b5d077)
#define FIX_LN_COEF_22 ((fix_internal) 0x01c513ef9d7bc3fc)
#define FIX_LN_COEF_23 ((fix_internal) 0x035089636a9b28c4)
#define FIX_LN_COEF_24 ((fix_internal) 0xfc97e4bb8417b13c)

#define FIX_LOG2_COEF_0 ((fix_internal) 0xfffffffffffffccd)
#define FIX_LOG2_COEF_1 ((fix_internal) 0x171547652b831bc8)
#define FIX_LOG2_COEF_2 ((fix_internal) 0xf4755c4d6a40ff21)
#define FIX_LOG2_COEF_3 ((fix_internal) 0x07b1c2770e6f95b1)
#define FIX_LOG2_COEF_4 ((fix_internal) 0xfa3aae26b327fd56)
#define FIX_LOG2_COEF_5 ((fix_internal) 0x049ddb14447f6649)
#define FIX_LOG2_COEF_6 ((fix_internal) 0xfc271ec529f2cc64)
#define FIX_LOG2_COEF_7 ((fix_internal) 0x034c2ec34219a06d)
#define FIX_LOG2_COEF_8 ((fix_internal) 0xfd1d56f2ee6d8bf6)
#define FIX_LOG2_COEF_9 ((fix_internal) 0x029096704dafbfb1)
#define FIX_LOG2_COEF_10 ((fix_internal) 0xfdb115f8a5fcceb2)
#define FIX_LOG2_COEF_11 ((fix_internal) 0x02192ea5b51aded8)
#define FIX_LOG2_COEF_12 ((fix_internal) 0xfe13515bc304c691)
#define FIX_LOG2_COEF_13 ((fix_internal) 0x01c6efd48f86a572)
#define FIX_LOG2_COEF_14 ((fix_internal) 0xfe5cc05f45cc5309)
#define FIX_LOG2_COEF_15 ((fix_internal) 0x01860764a098f63b)
#define FIX_LOG2_COEF_16 ((fix_internal) 0xfe781bdb29e8590a)
#define FIX_LOG2_COEF_17 ((fix_internal) 0x0177fc5e9ca545f1)
#define FIX_LOG2_COEF_18 ((fix_internal) 0xff308de3a5a8eef7)
#define FIX_LOG2_COEF_19 ((fix_internal) 0x00a996c619316599)
#define FIX_LOG2_COEF_20 ((fix_internal) 0xfd2f527f341f5144)
#define FIX_LOG2_COEF_21 ((fix_internal) 0x02ef6db52cb5a597)
#define FIX_LOG2_COEF_22 ((fix_internal) 0x0287c18807ce78ee)
#define FIX_LOG2_COEF_23 ((fix_internal) 0xfd39629a850abd71)
#define FIX_LOG2_COEF_24 ((fix_internal) 0xfb19a0db2c9bcd88)
#define FIX_LOG2_COEF_25 ((fix_internal) 0x04ec41d2e255a85d)

#define FIX_LOG10_COEF_0 ((fix_internal) 0xffffffffffffff9f)
#define FIX_LOG10_COEF_1 ((fix_internal) 0x06f2dec549b924cd)
#define FIX_LOG10_COEF_2 ((fix_internal) 0xfc86909d5b23d2ee)
#define FIX_LOG10_COEF_3 ((fix_internal) 0x0250f4ec6e12e84d)
#define FIX_LOG10_COEF_4 ((fix_internal) 0xfe43484eacf3eaa3)
#define FIX_LOG10_COEF_5 ((fix_internal) 0x0163c6276041092e)
#define FIX_LOG10_COEF_6 ((fix_internal) 0xfed7858a06123b09)
#define FIX_LOG10_COEF_7 ((fix_internal) 0x00fe1fd7c992de32)
#define FIX_LOG10_COEF_8 ((fix_internal) 0xff21a41c1a30d4be)
#define FIX_LOG10_COEF_9 ((fix_internal) 0x00c5a65facbc6d8f)
#define FIX_LOG10_COEF_10 ((fix_internal) 0xff4e1e1f187446f8)
#define FIX_LOG10_COEF_11 ((fix_internal) 0x00a1c31eeeef8ca1)
#define FIX_LOG10_COEF_12 ((fix_internal) 0xff6bae014b081460)
#define FIX_LOG10_COEF_13 ((fix_internal) 0x00883bba1a96f255)
#define FIX_LOG10_COEF_14 ((fix_internal) 0xff81df1711437907)
#define FIX_LOG10_COEF_15 ((fix_internal) 0x007bc41819356337)
#define FIX_LOG10_COEF_16 ((fix_internal) 0xff898a12d8caad6a)
#define FIX_LOG10_COEF_17 ((fix_internal) 0x004b03b357f2162d)
#define FIX_LOG10_COEF_18 ((fix_internal) 0xffc3939b703070ba)
#define FIX_LOG10_COEF_19 ((fix_internal) 0x00cc1ac9eaf0d78e)
#define FIX_LOG10_COEF_20 ((fix_internal) 0xff21c4363154f309)
#define FIX_LOG10_COEF_21 ((fix_internal) 0xff59bf71a04fea3f)
#define FIX_LOG10_COEF_22 ((fix_internal) 0x00cae92c577ed17d)
#define FIX_LOG10_COEF_23 ((fix_internal) 0x016ff17143ba55e4)
#define FIX_LOG10_COEF_24 ((fix_internal) 0xfe8138bb85ffe978)

#define CORDIC_N 33
#define CORDIC_P 0x9b74eda8435e5a6
#define CORDIC_LUT fix_internal cordic_lut[33] = { \
  0x0800000000000000,\
  0x04b90147677cc21a,\
  0x027ece16d7b8e7a3,\
  0x0144447507776687,\
  0x00a2c350c39626bb,\
  0x005175f85641189e,\
  0x0028bd87970a098a,\
  0x00145f15447510ac,\
  0x000a2f94d1b430ce,\
  0x000517cbaecc2ace,\
  0x00028be600246e9f,\
  0x000145f3052a032e,\
  0x0000a2f98337fb18,\
  0x0000517cc1b05cbd,\
  0x000028be60daba44,\
  0x0000145f306dae9f,\
  0x00000a2f9836e17f,\
  0x00000517cc1b7205,\
  0x0000028be60db92b,\
  0x00000145f306dc9b,\
  0x000000a2f9836e4e,\
  0x000000517cc1b727,\
  0x00000028be60db94,\
  0x000000145f306dca,\
  0x0000000a2f9836e5,\
  0x0000000517cc1b72,\
  0x000000028be60db9,\
  0x0000000145f306dd,\
  0x00000000a2f9836e,\
  0x00000000517cc1b7,\
  0x0000000028be60dc,\
  0x00000000145f306e,\
  0x000000000a2f9837\
};

#endif
