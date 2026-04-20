#pragma once

#include <cstddef>
#include <cstdint>

extern const float {{symbol}}_test_input[{{input_count}}];
extern const int8_t {{symbol}}_reference_output[{{output_count}}];
extern const float {{symbol}}_normalize_mean[{{channels}}];
extern const float {{symbol}}_normalize_std[{{channels}}];
extern const int {{symbol}}_test_input_index;
{{class_names_decl}}
{{target_decl}}
