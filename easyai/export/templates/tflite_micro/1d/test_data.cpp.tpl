#include "test_data.h"

const float {{symbol}}_test_input[{{input_count}}] = {
{{test_input_values}}
};

const float {{symbol}}_reference_output[{{output_count}}] = {
{{reference_output_values}}
};

{{target_def}}

const int {{symbol}}_test_input_index = {{sample_index}};
