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
    """
    Apply layer normalization to the input tensor.

    Parameters:
    - inputs (tf.Tensor): Input tensor.
    - name (str): Name for the layer (optional).

    Returns:
    - tf.Tensor: Normalized tensor.
    """
    norm_axis = -1 if K.image_data_format() == "channels_last" else 1
    return LayerNormalization(axis=norm_axis, epsilon=1e-5, name=name)(inputs)

def mlp_block(inputs, hidden_dim, activation="gelu", name=None):
    """
    Multilayer perceptron (MLP) block as defined in the paper.

    Parameters:
    - inputs (tf.Tensor): Input tensor.
    - hidden_dim (int): Dimension of the hidden layer.
    - activation (str): Activation function for the hidden layer (default is "gelu").
    - name (str): Name for the layer (optional).

    Returns:
    - tf.Tensor: Output tensor.
    """
    
    # 2 Dense layers with Gelu activation.
    nn = Dense(hidden_dim, name=name + "Dense_0")(inputs)
    nn = Activation(activation, name=name + "gelu")(nn)
    nn = Dense(inputs.shape[-1], name=name + "Dense_1")(nn)
    return nn

def mixer_block(inputs, tokens_mlp_dim, channels_mlp_dim, drop_rate=0, activation="gelu", name=None):
    """
    Mixer block. Architecture is a replica of the paper, and implementation is credited to https://github.com/leondgarse/keras_mlp/ . We needed to have the exact nomenclature as the model present in this repo so that we can load pretrained weights to be able to successfully fine-tune on any datasets.

    Parameters:
    - inputs (tf.Tensor): Input tensor.
    - tokens_mlp_dim (int): Dimension of the hidden layer for token mixing MLP.
    - channels_mlp_dim (int): Dimension of the hidden layer for channel mixing MLP.
    - drop_rate (float): Dropout rate (default is 0).
    - activation (str): Activation function for MLP blocks (default is "gelu").
    - name (str): Name for the layer (optional).

    Returns:
    - tf.Tensor: Output tensor.
    """
    # Token Mixing Block
    
    # Get a layer norm of input
    nn = layer_norm(inputs, name=name + "LayerNorm_0")
    
    # Transpose
    nn = Permute((2, 1), name=name + "permute_0")(nn)
    
    # Insert a whole MLP Block
    nn = mlp_block(nn, tokens_mlp_dim, activation, name=name + "token_mixing/")
    
    # Transpose
    nn = Permute((2, 1), name=name + "permute_1")(nn)
    
    # Include Dropout 
    if drop_rate > 0:
        nn = Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "token_drop")(nn)
        
    # Residual connection
    token_out = Add(name=name + "add_0")([nn, inputs])

    # Channel Mixing Block
    # layer Norm of output of token mixing block
    nn = layer_norm(token_out, name=name + "LayerNorm_1")
    
    # MLP Block as before
    channel_out = mlp_block(nn, channels_mlp_dim, activation, name=name + "channel_mixing/")
    
    # Dropout
    if drop_rate > 0:
        channel_out = Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "channel_drop")(channel_out)

    # Final output with residual connection
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
    """
    Create an MLP Mixer model. Architecture is a replica of the paper, and implementation is credited to https://github.com/leondgarse/keras_mlp/ . We needed to have the exact nomenclature as the model present in this repo so that we can load pretrained weights to be able to successfully fine-tune on any datasets.

    Parameters:
    - num_blocks (int): Number of Mixer blocks in the model.
    - patch_size (int): Size of the patch in the stem convolution.
    - stem_width (int): Width of the stem convolution.
    - tokens_mlp_dim (int): Dimension of the hidden layer for token mixing MLP in Mixer blocks.
    - channels_mlp_dim (int): Dimension of the hidden layer for channel mixing MLP in Mixer blocks.
    - input_shape (tuple): Shape of the input image (default is (224, 224, 3)).
    - num_classes (int): Number of output classes (default is 0 for feature extraction).
    - dropout (float): Dropout rate for the final classifier layer (default is 0).
    - drop_connect_rate (float or list/tuple): Drop connect rate for skip connections in Mixer blocks.
    - initial_activation (str): Activation function for the initial stem convolution (default is "relu").
    - mixer_activation (str): Activation function for the MLP blocks in Mixer blocks (default is "gelu").
    - classifier_activation (str): Activation function for the final classifier layer (default is "softmax").
    - model_name (str): Name for the model (default is "mlp_mixer").
    - pretrained (str): Pretrained model name for loading weights (optional).
    - local_model (str): Path to a locally saved model to load instead of creating a new one (optional).
    - url (str): URL to load pretrained weights (optional).
    - unfreeze (str): Specifies which layers to unfreeze ("top" for only the top classifier layer, "all" for all layers).

    Returns:
    - tf.keras.Model: MLP Mixer model.
    """
    
    # Input layer
    inputs = keras.Input(input_shape)
    
    # A "Conv" Layer, but it is equivalent to a fully connected layer operating on each patch separately, since we have stride and kernel size same as the patch size, and the kernal share weights. 
    nn = Conv2D(stem_width, kernel_size=patch_size, strides=patch_size, padding="same", name="stem",
                             activation=initial_activation if initial_activation else None)(inputs)
    
    # Reshape to form linear Embeddings
    nn = Reshape([nn.shape[1] * nn.shape[2], stem_width])(nn)
    
    # Drop connect rates if defined
    drop_connect_s, drop_connect_e = drop_connect_rate if isinstance(drop_connect_rate, (list, tuple)) else [
        drop_connect_rate, drop_connect_rate]
    
    # Create the num_blocks specified according to the architecture
    for ii in range(num_blocks):
        name = f"MixerBlock_{str(ii)}"
        block_drop_rate = drop_connect_s + (drop_connect_e - drop_connect_s) * ii / num_blocks
        
        # Create the Mixer block
        nn = mixer_block(nn, tokens_mlp_dim, channels_mlp_dim, drop_rate=block_drop_rate, activation=mixer_activation,
                         name=name)
        
    # Layer normalization before the head    
    nn = layer_norm(nn, name="pre_head_layer_norm")

    # Classification head 
    if num_classes > 0:
        # Equivalent to reduce_mean along axis 1
        nn = GlobalAveragePooling1D()(nn)  
        
        # Dropout if defined
        if dropout > 0 and dropout < 1:
            nn = Dropout(dropout)(nn)
            
        # Final dense layer with the classifier activation
        nn = Dense(num_classes, activation=classifier_activation, name="head")(nn)
    
    # Create the whole model using Keras's functional technique
    model = keras.Model(inputs, nn, name=model_name)

    # Model loading and freezing/unfreezing   
    if not local_model:
        # If there is no local model specified and we have a pretrained model
        if pretrained:
            
            # Try loading weights into the model from the specification
            reload_model_weights(model, f"{url}/{pretrained}.h5")
            print(">>>> Loaded Pretrained Model Successfully !")
            
            # If top layers to unfreeze, 
            if unfreeze == "top":
                # first unfreeze everything to make sure, 
                for layer in model.layers:
                    layer.trainable = True
                # and then freeze everything before the classifier head
                for layer in model.layers[:-3]:
                    layer.trainable = False
                print(">>>> Only Top Classifier Layer is unfrozen")
            else:
                # For None or any other string than top, just unfreeze everything
                for layer in model.layers:
                    layer.trainable = True
                print(">>>> All Layers are trainable")
        else:
            print("Loaded Model Successfully without any pretrained or saved weights")
            return model
    else:
        # if local model, try loading it directly.
        model = load_model(f"{local_model}.h5")
        print(">>>> Loaded Locally Saved Model Successfully !")
    
    return model


def reload_model_weights(model, url=""):
    """
    Load weights for a given model from a specific pretrained directory.
    Credits for providing pretrained models: https://github.com/leondgarse/keras_mlp/
    
    Parameters:
    - model (tf.keras.Model): The model for which weights are to be loaded.
    - url (str): URL pointing to the location of the pretrained weights.

    """
    
    # Extract the file name from the URL
    file_name = os.path.basename(url)

    try:
        # Attempt to download the pretrained weights file
        pretrained_model = keras.utils.get_file(file_name, url, cache_subdir="models")
    except:
        # If download fails
        print("COULD NOT LOAD WEIGHTS. PLEASE RE-CHECK THE URL AND PRETRAINED MODEL NAME")
        return
    else:
        # Load weights into the model, skipping layers with mismatched shapes
        model.load_weights(pretrained_model, by_name=True, skip_mismatch=True)
        print("\nSuccessfully Loaded pretrained model from : ", pretrained_model)
