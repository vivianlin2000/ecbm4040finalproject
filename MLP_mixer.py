#!/usr/bin/env python
# coding: utf-8

import os, warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.experimental import CosineDecay
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.layers import (
    Add,
    Dense,
    Conv2D,
    GlobalAveragePooling1D,
    Layer,
    LayerNormalization,
    Permute,
    Softmax,
    Activation,
)


# Reproducability
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


set_seed()

from util_functions import reload_model_weights


def layer_norm(inputs, name=None):
    BATCH_NORM_EPSILON = 1e-5
    norm_axis = -1 if K.image_data_format() == "channels_last" else 1
    return keras.layers.LayerNormalization(axis=norm_axis, epsilon=BATCH_NORM_EPSILON, name=name)(inputs)


def mlp_block(inputs, hidden_dim, activation="gelu", name=None):
    nn = keras.layers.Dense(hidden_dim, name=name + "Dense_0")(inputs)
    nn = keras.layers.Activation(activation, name=name + "gelu")(nn)
    nn = keras.layers.Dense(inputs.shape[-1], name=name + "Dense_1")(nn)
    return nn


def mixer_block(inputs, tokens_mlp_dim, channels_mlp_dim, drop_rate=0, activation="gelu", name=None):
    nn = layer_norm(inputs, name=name + "LayerNorm_0")
    nn = keras.layers.Permute((2, 1), name=name + "permute_0")(nn)
    nn = mlp_block(nn, tokens_mlp_dim, activation, name=name + "token_mixing/")
    nn = keras.layers.Permute((2, 1), name=name + "permute_1")(nn)
    if drop_rate > 0:
        nn = keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "token_drop")(nn)
    token_out = keras.layers.Add(name=name + "add_0")([nn, inputs])

    nn = layer_norm(token_out, name=name + "LayerNorm_1")
    channel_out = mlp_block(nn, channels_mlp_dim, activation, name=name + "channel_mixing/")
    if drop_rate > 0:
        channel_out = keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "channel_drop")(channel_out)
    return keras.layers.Add(name=name + "add_1")([channel_out, token_out])


def MLPMixer(
        num_blocks,
        patch_size,
        stem_width,
        tokens_mlp_dim,
        channels_mlp_dim,
        input_shape=(224, 224, 3),
        num_classes=0,
        dropout=0,
        drop_connect_rate=0,
        initial_activation="relu",
        mixer_activation="gelu",
        classifier_activation="softmax",
        model_name="mlp_mixer",
        pretrained=None,
        local_model=False,
        url=None,
        unfreeze="top"
):
    inputs = keras.Input(input_shape)
    nn = keras.layers.Conv2D(stem_width, kernel_size=patch_size, strides=patch_size, padding="same", name="stem",
                             activation=initial_activation if initial_activation else None)(inputs)
    nn = keras.layers.Reshape([nn.shape[1] * nn.shape[2], stem_width])(nn)

    drop_connect_s, drop_connect_e = drop_connect_rate if isinstance(drop_connect_rate, (list, tuple)) else [
        drop_connect_rate, drop_connect_rate]
    for ii in range(num_blocks):
        name = "{}_{}/".format("MixerBlock", str(ii))
        block_drop_rate = drop_connect_s + (drop_connect_e - drop_connect_s) * ii / num_blocks
        nn = mixer_block(nn, tokens_mlp_dim, channels_mlp_dim, drop_rate=block_drop_rate, activation=mixer_activation,
                         name=name)
    nn = layer_norm(nn, name="pre_head_layer_norm")

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling1D()(nn)  # tf.reduce_mean(nn, axis=1)
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Dense(num_classes, activation=classifier_activation, name="head")(nn)

    model = keras.Model(inputs, nn, name=model_name)

    if not local_model:
        if pretrained:
            reload_model_weights(model, input_shape, "{}/{}.h5".format(url, pretrained))
            print(">>>> Loaded Pretrained Model")
            if unfreeze == "top":
                for layer in model.layers:
                    layer.trainable = True
                for layer in model.layers[:-3]:
                    layer.trainable = False
                print(">>>> Unfroze Top Layers")
            else:
                for layer in model.layers:
                    layer.trainable = True
                print(">>>> Unfroze All Layers")
        else:
            return model
    else:
        model = load_model("{}.h5".format(local_model))
        print(">>>> Loaded Locally Saved Model")
    return model
