#pragma once

#include <cstddef>
#include <cstdint>

bool {{symbol}}_model_init();
bool {{symbol}}_model_predict(const float* input, size_t input_len, int8_t* output, size_t output_len);
{{label_decl}}
