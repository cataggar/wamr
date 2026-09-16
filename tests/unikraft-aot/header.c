#include "wamr_aot.h"

_Static_assert(sizeof(wamr_aot_value) == 16, "value ABI");
_Static_assert(offsetof(wamr_aot_value, bits) == 8, "value bits ABI");
_Static_assert(sizeof(wamr_aot_result) == 24, "result ABI");
_Static_assert(offsetof(wamr_aot_result, error_name) == 16, "error name ABI");
_Static_assert(sizeof(wamr_aot_config) == 72, "platform configuration ABI");
_Static_assert(sizeof(wamr_aot_import) == 80, "import ABI");

wamr_aot_result header_call(wamr_aot_handle *instance)
{
    const wamr_aot_value args[] = {
        { WAMR_AOT_I32, 0, 20 }, { WAMR_AOT_I32, 0, 22 }
    };
    wamr_aot_value result;
    return wamr_aot_call(instance, (const uint8_t *)"add", 3,
                         args, 2, &result, 1);
}
