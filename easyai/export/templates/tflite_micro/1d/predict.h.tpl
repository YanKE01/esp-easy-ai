#pragma once

#include <stddef.h>

bool {{symbol}}_model_init();
bool {{symbol}}_model_predict(
    const float* input,
    size_t input_len,
    float* output,
    size_t output_len);
{{label_decl}}
