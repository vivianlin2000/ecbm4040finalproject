#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.layers import (
    Add,
    Dense,
    Conv2D,
    Dropout,
    GlobalAveragePooling1D,
    LayerNormalization,
    Permute,
    Reshape,
    Activation,
)


# Reproducability
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


set_seed()




def layer_norm(inputs, name=None):
    BATCH_NORM_EPSILON = 1e-5
    norm_axis = -1 if K.image_data_format() == "channels_last" else 1
    return LayerNormalization(axis=norm_axis, epsilon=BATCH_NORM_EPSILON, name=name)(inputs)


def mlp_block(inputs, hidden_dim, activation="gelu", name=None):
    nn = Dense(hidden_dim, name=name + "Dense_0")(inputs)
    nn = Activation(activation, name=name + "gelu")(nn)
    nn = Dense(inputs.shape[-1], name=name + "Dense_1")(nn)
    return nn


def mixer_block(inputs, tokens_mlp_dim, channels_mlp_dim, drop_rate=0, activation="gelu", name=None):
    nn = layer_norm(inputs, name=name + "LayerNorm_0")
    nn = Permute((2, 1), name=name + "permute_0")(nn)
    nn = mlp_block(nn, tokens_mlp_dim, activation, name=name + "token_mixing/")
    nn = Permute((2, 1), name=name + "permute_1")(nn)
    if drop_rate > 0:
        nn = Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "token_drop")(nn)
    token_out = Add(name=name + "add_0")([nn, inputs])

    nn = layer_norm(token_out, name=name + "LayerNorm_1")
    channel_out = mlp_block(nn, channels_mlp_dim, activation, name=name + "channel_mixing/")
    if drop_rate > 0:
        channel_out = Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "channel_drop")(channel_out)
    return Add(name=name + "add_1")([channel_out, token_out])


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
    nn = Conv2D(stem_width, kernel_size=patch_size, strides=patch_size, padding="same", name="stem",
                             activation=initial_activation if initial_activation else None)(inputs)
    nn = Reshape([nn.shape[1] * nn.shape[2], stem_width])(nn)

    drop_connect_s, drop_connect_e = drop_connect_rate if isinstance(drop_connect_rate, (list, tuple)) else [
        drop_connect_rate, drop_connect_rate]
    
    for ii in range(num_blocks):
        name = f"MixerBlock_{str(ii)}"
        block_drop_rate = drop_connect_s + (drop_connect_e - drop_connect_s) * ii / num_blocks
        nn = mixer_block(nn, tokens_mlp_dim, channels_mlp_dim, drop_rate=block_drop_rate, activation=mixer_activation,
                         name=name)
        
    nn = layer_norm(nn, name="pre_head_layer_norm")

    if num_classes > 0:
        nn = GlobalAveragePooling1D()(nn)  # tf.reduce_mean(nn, axis=1)
        if dropout > 0 and dropout < 1:
            nn = Dropout(dropout)(nn)
        nn = Dense(num_classes, activation=classifier_activation, name="head")(nn)

    model = keras.Model(inputs, nn, name=model_name)

    if not local_model:
        if pretrained:
            reload_model_weights(model, f"{url}/{pretrained}.h5")
            print(">>>> Loaded Pretrained Model Successfully !")

            if unfreeze == "top":
                for layer in model.layers:
                    layer.trainable = True
                for layer in model.layers[:-3]:
                    layer.trainable = False
                print(">>>> Only Top Classifier Layer is unfrozen")
            else:
                for layer in model.layers:
                    layer.trainable = True
                print(">>>> All Layers are trainable")
        else:
            print("Loaded Model Successfully without any pretrained or saved weights")
            return model
    else:
        model = load_model(f"{local_model}.h5")
        print(">>>> Loaded Locally Saved Model Successfully !")
    
    return model


def reload_model_weights(model, url=""):

    file_name = os.path.basename(url)

    try:
        pretrained_model = keras.utils.get_file(file_name, url, cache_subdir="models")
    except:
        print("COULD NOT LOAD WEIGHTS. PLEASE RE-CHECK THE URL AND PRETRAUNED MODEL NAME:", url)
        return
    else:
        model.load_weights(pretrained_model, by_name=True, skip_mismatch=True)
        print("\nSuccessfully Loaded pretrained model from : ", pretrained_model)
