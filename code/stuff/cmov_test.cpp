#include <stdint.h>
#include <cstdlib>
#include <stdio.h>


/* move new_val to dst if pred is true */
uint32_t cmov(uint8_t pred, uint32_t target_val, uint32_t new_val)
{
    uint32_t result;
    __asm__ volatile (
        "mov %2, %0;\n\t"
        "test %1, %1;\n\t"
        "cmovz %3, %0;\n\t"
        "test %2, %2;"
        : "=&r"(result)  // need the & early-clobber
        : "r"(pred), "r"(target_val), "r"(new_val)
        : "cc");
    return result;
}

int main(int argc, char *argv[]){

  uint8_t pred = atoi(argv[1]);
  uint32_t val = atoi(argv[2]);
  uint32_t target = atoi(argv[3]);

  uint32_t res = cmov(pred, target, val);

  printf("result: %i", res);

}
