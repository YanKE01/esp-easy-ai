#include "predict.h"

#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"
#include "test_data.h"

extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");

namespace {

dl::Model* g_model = nullptr;
dl::TensorBase* g_model_input = nullptr;
dl::TensorBase* g_model_output = nullptr;
bool g_initialized = false;

}  // namespace

static inline size_t SourceFlatIndex(size_t channel, size_t timestep) {
    return channel * {{raw_input_length}} + timestep;
}

static inline float ApplyPreprocess(float value, size_t channel) {
    value *= {{rescale_literal}};
    value = (value - {{symbol}}_normalize_mean[channel]) / {{symbol}}_normalize_std[channel];
    return value;
}

bool {{symbol}}_model_init() {
    if (g_initialized) {
        return true;
    }

    g_model = new dl::Model(reinterpret_cast<const char*>(model_espdl), fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    if (g_model == nullptr) {
        return false;
    }

    auto inputs = g_model->get_inputs();
    auto outputs = g_model->get_outputs();
    if (inputs.empty() || outputs.empty()) {
        return false;
    }

    g_model_input = inputs.begin()->second;
    g_model_output = outputs.begin()->second;
    g_initialized = true;
    return true;
}

bool {{symbol}}_model_predict(const float* input, size_t input_len, int8_t* output, size_t output_len) {
    if (!g_initialized && !{{symbol}}_model_init()) {
        return false;
    }
    if (input == nullptr || output == nullptr) {
        return false;
    }
    if (input_len != {{raw_input_count}}) {
        return false;
    }
    if (output_len != {{output_count}}) {
        return false;
    }

    // The exported raw test_input keeps training/config semantics [C, L].
    // The current deployed ESP-DL tensor layout for 1D models is [N, L, C].
    // The loops below therefore write input data in NLC order when required.
    if (g_model_input->get_dtype() == dl::DATA_TYPE_INT8) {
        const float input_inv_scale = DL_RESCALE(g_model_input->get_exponent());
        int8_t* input_ptr = g_model_input->get_element_ptr<int8_t>();
        size_t dst = 0;
        {{quantize_loop}}
    } else if (g_model_input->get_dtype() == dl::DATA_TYPE_FLOAT) {
        float* input_ptr = g_model_input->get_element_ptr<float>();
        size_t dst = 0;
        {{float_input_loop}}
    } else {
        return false;
    }

    g_model->run();

    if (g_model_output->get_dtype() != dl::DATA_TYPE_INT8) {
        return false;
    }

    int8_t* output_ptr = g_model_output->get_element_ptr<int8_t>();
    for (size_t i = 0; i < output_len; ++i) {
        output[i] = output_ptr[i];
    }
    return true;
}
{{label_impl}}

// Project: {{symbol}}
// Export backend: esp-dl
// Raw input shape [C, L]: {{raw_input_shape}}
// Training/config semantic layout: NCL
// Current deployed ESP-DL layout for 1D export: NLC
// Model input tensor: {{input_tensor_name}}, shape={{model_input_shape}}, exponent={{input_exponent}}
// Output tensor: {{output_tensor_name}}, exponent={{output_exponent}}
// Exported reference sample index: {{sample_index}}
// test_input is raw float data. Preprocess, int8 quantization, and layout adaptation are applied inside model_predict().
{{target_comment}}
