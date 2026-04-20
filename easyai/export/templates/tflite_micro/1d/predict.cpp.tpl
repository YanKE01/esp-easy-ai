#include "predict.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"

namespace {

constexpr size_t kInputChannels = {{input_channels}};
constexpr size_t kInputLength = {{input_length}};
constexpr size_t kInputCount = {{input_count}};
constexpr size_t kOutputCount = {{output_count}};
constexpr size_t kTensorArenaSize = {{tensor_arena_size}};

alignas(16) uint8_t g_tensor_arena[kTensorArenaSize];
tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input_tensor = nullptr;
TfLiteTensor* g_output_tensor = nullptr;
bool g_initialized = false;

const float g_normalize_mean[kInputChannels] = {
{{normalize_mean_values}}
};

const float g_normalize_std[kInputChannels] = {
{{normalize_std_values}}
};

inline float ApplyPreprocess(float value, size_t channel) {
    value *= {{rescale_literal}};
    return (value - g_normalize_mean[channel]) / g_normalize_std[channel];
}

inline int8_t QuantizeInt8(float value, float scale, int zero_point) {
    if (scale == 0.0f) {
        return static_cast<int8_t>(zero_point);
    }
    const float transformed = value / scale + static_cast<float>(zero_point);
    int quantized = static_cast<int>(roundf(transformed));
    if (quantized < -128) quantized = -128;
    if (quantized > 127) quantized = 127;
    return static_cast<int8_t>(quantized);
}

inline size_t SourceFlatIndex(size_t channel, size_t timestep) {
    return channel * kInputLength + timestep;
}

}  // namespace

bool {{symbol}}_model_init() {
    if (g_initialized) {
        return true;
    }

    tflite::InitializeTarget();
    const tflite::Model* model = tflite::GetModel({{symbol}}_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model schema mismatch: %d != %d", model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    static tflite::MicroMutableOpResolver<{{resolver_size}}> resolver;
{{resolver_code}}

    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        g_tensor_arena,
        kTensorArenaSize);
    g_interpreter = &static_interpreter;

    if (g_interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return false;
    }

    g_input_tensor = g_interpreter->input(0);
    g_output_tensor = g_interpreter->output(0);
    g_initialized = true;
    return true;
}

bool {{symbol}}_model_predict(
    const float* input,
    size_t input_len,
    float* output,
    size_t output_len) {
    if (!g_initialized && !{{symbol}}_model_init()) {
        return false;
    }
    if (input == nullptr || output == nullptr) {
        return false;
    }
    if (input_len != kInputCount || output_len != kOutputCount) {
        return false;
    }

    if ({{input_is_int8}}) {
        int8_t* input_ptr = g_input_tensor->data.int8;
        size_t dst = 0;
        for (size_t t = 0; t < kInputLength; ++t) {
            for (size_t c = 0; c < kInputChannels; ++c) {
                const size_t src = {{reorder_to_lc}} ? SourceFlatIndex(c, t) : dst;
                const float normalized = ApplyPreprocess(input[src], c);
                input_ptr[dst++] = QuantizeInt8(
                    normalized,
                    {{input_scale}},
                    {{input_zero_point}});
            }
        }
    } else {
        float* input_ptr = g_input_tensor->data.f;
        size_t dst = 0;
        for (size_t t = 0; t < kInputLength; ++t) {
            for (size_t c = 0; c < kInputChannels; ++c) {
                const size_t src = {{reorder_to_lc}} ? SourceFlatIndex(c, t) : dst;
                input_ptr[dst++] = ApplyPreprocess(input[src], c);
            }
        }
    }

    if (g_interpreter->Invoke() != kTfLiteOk) {
        MicroPrintf("Invoke failed");
        return false;
    }

    if ({{output_is_int8}}) {
        const int8_t* output_ptr = g_output_tensor->data.int8;
        for (size_t i = 0; i < output_len; ++i) {
            output[i] = (static_cast<int>(output_ptr[i]) - {{output_zero_point}}) * {{output_scale}};
        }
    } else {
        memcpy(output, g_output_tensor->data.f, output_len * sizeof(float));
    }
    return true;
}

{{label_impl}}

// Project: {{symbol}}
// Export backend: tflite-micro
// Exported reference sample index: {{sample_index}}
// Raw test_input keeps training/config semantics [C, L].
// Current TFLite Micro 1D path feeds deployed input as [L, C].
{{target_comment}}
