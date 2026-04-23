#pragma once

#include <stddef.h>

extern const float {{symbol}}_test_input[{{input_count}}];
extern const float {{symbol}}_reference_output[{{output_count}}];
{{target_decl}}
extern const int {{symbol}}_test_input_index;
