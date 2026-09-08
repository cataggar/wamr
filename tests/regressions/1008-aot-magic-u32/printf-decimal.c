#include <stdint.h>
#include <stdio.h>

static volatile uint32_t inputs[] = {
    1385676899u,
    2147483648u,
    3000000008u,
};

int main(void)
{
    printf("{\"checksum\":%llu,\"cases\":[",
           (unsigned long long)13856768990818897060ull);
    for (unsigned i = 0; i < sizeof(inputs) / sizeof(inputs[0]); ++i) {
        uint32_t value = inputs[i];
        printf("%s[%u,%u,%u]", i == 0 ? "" : ",",
               value, value / 10u, value % 10u);
    }
    puts("]}");
    return 0;
}
