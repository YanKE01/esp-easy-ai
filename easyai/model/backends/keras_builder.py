"""Keras model builder for the unified layer DSL."""

from __future__ import annotations

from easyai.config.schema import ModelConfig, TaskConfig
from easyai.model.errors import ModelTranslationError


SUPPORTED_KERAS_LAYERS = {
    "Conv1d",
    "Conv2d",
    "DepthwiseConv2d",
    "Linear",
    "BatchNorm1d",
    "BatchNorm2d",
    "MaxPool1d",
    "MaxPool2d",
    "AvgPool1d",
    "AvgPool2d",
    "GlobalAvgPool1d",
    "GlobalAvgPool2d",
    "Flatten",
    "Dropout",
    "ReLU",
    "ReLU6",
    "LeakyReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
}


def _layer_kwargs(layer):
    return layer.model_dump(exclude={"type"})


def _keras():
    import tensorflow as tf

    return tf


def _resolve_stride(kwargs, kernel_size, key="stride"):
    stride = kwargs.pop(key, None)
    return stride if stride is not None else kernel_size


def _maybe_add_padding_1d(layers_out, padding):
    if padding:
        layers_out.append(_keras().keras.layers.ZeroPadding1D(padding=padding))


def _maybe_add_padding_2d(layers_out, padding):
    if padding:
        layers_out.append(_keras().keras.layers.ZeroPadding2D(padding=padding))


def _build_keras_layers(layer):
    tf = _keras()
    kwargs = _layer_kwargs(layer)
    layers_out = []

    if layer.type == "Conv1d":
        padding = kwargs.pop("padding", 0)
        stride = kwargs.pop("stride", 1)
        kwargs.pop("in_channels", None)
        _maybe_add_padding_1d(layers_out, padding)
        layers_out.append(
            tf.keras.layers.Conv1D(
                filters=kwargs.pop("out_channels"),
                kernel_size=kwargs.pop("kernel_size"),
                strides=stride,
                padding="valid",
                **kwargs,
            )
        )
        return layers_out
    if layer.type == "Conv2d":
        padding = kwargs.pop("padding", 0)
        stride = kwargs.pop("stride", 1)
        kwargs.pop("in_channels", None)
        _maybe_add_padding_2d(layers_out, padding)
        layers_out.append(
            tf.keras.layers.Conv2D(
                filters=kwargs.pop("out_channels"),
                kernel_size=kwargs.pop("kernel_size"),
                strides=stride,
                padding="valid",
                **kwargs,
            )
        )
        return layers_out
    if layer.type == "DepthwiseConv2d":
        padding = kwargs.pop("padding", 0)
        stride = kwargs.pop("stride", 1)
        kwargs.pop("in_channels", None)
        _maybe_add_padding_2d(layers_out, padding)
        layers_out.append(
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=kwargs.pop("kernel_size"),
                strides=stride,
                padding="valid",
                **kwargs,
            )
        )
        return layers_out
    if layer.type == "Linear":
        kwargs.pop("in_features", None)
        return [tf.keras.layers.Dense(units=kwargs.pop("out_features"), **kwargs)]
    if layer.type == "BatchNorm1d":
        return [tf.keras.layers.BatchNormalization(axis=-1)]
    if layer.type == "BatchNorm2d":
        return [tf.keras.layers.BatchNormalization(axis=-1)]
    if layer.type == "MaxPool1d":
        padding = kwargs.pop("padding", 0)
        kernel_size = kwargs.pop("kernel_size")
        _maybe_add_padding_1d(layers_out, padding)
        layers_out.append(
            tf.keras.layers.MaxPooling1D(
                pool_size=kernel_size,
                strides=_resolve_stride(kwargs, kernel_size),
                padding="valid",
                **kwargs,
            )
        )
        return layers_out
    if layer.type == "MaxPool2d":
        padding = kwargs.pop("padding", 0)
        kernel_size = kwargs.pop("kernel_size")
        _maybe_add_padding_2d(layers_out, padding)
        layers_out.append(
            tf.keras.layers.MaxPooling2D(
                pool_size=kernel_size,
                strides=_resolve_stride(kwargs, kernel_size),
                padding="valid",
                **kwargs,
            )
        )
        return layers_out
    if layer.type == "AvgPool1d":
        padding = kwargs.pop("padding", 0)
        kernel_size = kwargs.pop("kernel_size")
        _maybe_add_padding_1d(layers_out, padding)
        layers_out.append(
            tf.keras.layers.AveragePooling1D(
                pool_size=kernel_size,
                strides=_resolve_stride(kwargs, kernel_size),
                padding="valid",
                **kwargs,
            )
        )
        return layers_out
    if layer.type == "AvgPool2d":
        padding = kwargs.pop("padding", 0)
        kernel_size = kwargs.pop("kernel_size")
        _maybe_add_padding_2d(layers_out, padding)
        layers_out.append(
            tf.keras.layers.AveragePooling2D(
                pool_size=kernel_size,
                strides=_resolve_stride(kwargs, kernel_size),
                padding="valid",
                **kwargs,
            )
        )
        return layers_out
    if layer.type == "GlobalAvgPool1d":
        return [tf.keras.layers.GlobalAveragePooling1D()]
    if layer.type == "GlobalAvgPool2d":
        return [tf.keras.layers.GlobalAveragePooling2D()]
    if layer.type == "Flatten":
        return [tf.keras.layers.Flatten()]
    if layer.type == "Dropout":
        return [tf.keras.layers.Dropout(rate=kwargs.pop("p"), **kwargs)]
    if layer.type == "ReLU":
        return [tf.keras.layers.ReLU()]
    if layer.type == "ReLU6":
        return [tf.keras.layers.ReLU(max_value=6.0)]
    if layer.type == "LeakyReLU":
        slope = kwargs.pop("negative_slope", 0.01)
        return [tf.keras.layers.LeakyReLU(negative_slope=slope, **kwargs)]
    if layer.type == "Sigmoid":
        return [tf.keras.layers.Activation("sigmoid")]
    if layer.type == "Tanh":
        return [tf.keras.layers.Activation("tanh")]
    if layer.type == "Softmax":
        axis = kwargs.pop("dim", -1)
        return [tf.keras.layers.Softmax(axis=axis, **kwargs)]

    raise ModelTranslationError(f"Unsupported keras layer type: {layer.type}")


def build_keras_model(model_cfg: ModelConfig, task_cfg: TaskConfig):
    """Translate unified layers into a keras model."""
    unsupported = [
        layer.type
        for layer in model_cfg.layers
        if layer.type not in SUPPORTED_KERAS_LAYERS
    ]
    if unsupported:
        raise ModelTranslationError(
            f"Unsupported keras layer types: {sorted(set(unsupported))}"
        )

    keras_layers = []
    for layer in model_cfg.layers:
        keras_layers.extend(_build_keras_layers(layer))
    return _keras().keras.Sequential(keras_layers)
