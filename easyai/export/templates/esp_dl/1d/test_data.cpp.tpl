#include "test_data.h"

const float {{symbol}}_test_input[{{input_count}}] = {
{{test_input_values}}
};

const int8_t {{symbol}}_reference_output[{{output_count}}] = {
{{reference_output_values}}
};

const float {{symbol}}_normalize_mean[{{channels}}] = {
{{normalize_mean_values}}
};

const float {{symbol}}_normalize_std[{{channels}}] = {
{{normalize_std_values}}
};

const int {{symbol}}_test_input_index = {{sample_index}};
{{class_names_def}}
{{target_def}}
