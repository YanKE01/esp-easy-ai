#include <stdio.h>

#include "predict.h"
#include "test_data.h"

extern "C" void app_main(void)
{
    if (!{{symbol}}_model_init()) {
        printf("Model init failed.\n");
        return;
    }

{{main_body}}
}
