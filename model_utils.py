import os
from tensorflow import keras
from MLP_mixer import MLPMixer
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.experimental import CosineDecay
from tensorflow.keras.models import load_model




def get_model_architecture(model_name="b16"):
    MODEL_CONFIGS = {
        "s32": {
            "num_blocks": 8,
            "patch_size": 32,
            "stem_width": 512,
            "tokens_mlp_dim": 256,
            "channels_mlp_dim": 2048,
        },
        "s16": {
            "num_blocks": 8,
            "patch_size": 16,
            "stem_width": 512,
            "tokens_mlp_dim": 256,
            "channels_mlp_dim": 2048,
        },
        "b32": {
            "num_blocks": 12,
            "patch_size": 32,
            "stem_width": 768,
            "tokens_mlp_dim": 384,
            "channels_mlp_dim": 3072,
        },
        "b16": {
            "num_blocks": 12,
            "patch_size": 16,
            "stem_width": 768,
            "tokens_mlp_dim": 384,
            "channels_mlp_dim": 3072,
        },
        "l32": {
            "num_blocks": 24,
            "patch_size": 32,
            "stem_width": 1024,
            "tokens_mlp_dim": 512,
            "channels_mlp_dim": 4096,
        },
        "l16": {
            "num_blocks": 24,
            "patch_size": 16,
            "stem_width": 1024,
            "tokens_mlp_dim": 512,
            "channels_mlp_dim": 4096,
        },
    }

    return MODEL_CONFIGS[model_name]




def get_resolution(pretrained, dataset):
    if pretrained:
        return (224, 224, 3)
    else:
        if "cifar" in dataset.lower():
            return (32, 32, 3)
        else:
            return (224, 224, 3)


def get_num_classes(dataset):
    if dataset.lower() == "cifar10":
        return 10
    elif dataset.lower() == "flowers":
        return 102
    elif dataset.lower() == "pets":
        return 37
    elif dataset.lower() == "tiny-imagenet":
        return 200
    else:
        print("UNSUPPORTED DATASET. PLEASE CHECK NAME OF DATASET IN CONFIGS YOU PROVIDE.")


# this function is only for retrieving one of the 4 pretrained models from the git repo or create a brand new untrained model
def get_model(configs):
    local_model = configs["local_model_path"]
    if not local_model:
        num_blocks, patch_size, stem_width, tokens_mlp_dim, channels_mlp_dim = get_model_architecture(
            configs["model_name"]).values()

        model = MLPMixer(
            num_blocks=num_blocks,
            patch_size=patch_size,
            stem_width=stem_width,
            tokens_mlp_dim=tokens_mlp_dim,
            channels_mlp_dim=channels_mlp_dim,
            input_shape=get_resolution(configs["pretrained"], configs["dataset_name"]),
            num_classes=get_num_classes(configs["dataset_name"]),
            dropout=configs["dropout"],
            drop_connect_rate=configs["drop_connect_rate"],
            initial_activation=configs["initial_activation"],
            mixer_activation="gelu",
            classifier_activation="softmax",
            model_name="mlp_mixer",
            pretrained=configs["pretrained"],
            local_model=configs["local_model_path"],
            url='https://github.com/leondgarse/keras_mlp/releases/download/mlp_mixer/',
            unfreeze=configs["trainable_layers"]
        )
    else:
        model = load_model(f"{local_model}.h5")
        print(">>>> Loaded Locally Saved Model Successfully !")
    
    return model


def generate_model_configs(pretrained=None,
                           model_name="b16",
                           dataset_name="cifar10",
                           initial_activation="relu",
                           local_model_path=None,
                           trainable_layers="top",
                           optimizer_name = "SGD",
                           learning_rate = 0.001,
                           num_epochs = 10,
                           momentum = 0.9,
                           decay = 0.001,
                           cosine_decay = False,
                           batch_size = 32,
                           dropout = 0,
                           drop_connect_rate=0,
                           train_augmentation = True
                           ):
    
    
    return {"pretrained": pretrained,
            "model_name":model_name,
            "dataset_name": dataset_name,
            "initial_activation": initial_activation,
            "local_model_path": local_model_path,
            "learning_rate" : learning_rate,
            "trainable_layers": trainable_layers,
            "num_epochs" :num_epochs,
            "momentum" : momentum,
            "decay" :decay,
            "optimizer_name" : optimizer_name,
            "cosine_decay" : cosine_decay,
            "batch_size": batch_size,
            "dropout" : dropout,
            "drop_connect_rate" : drop_connect_rate,
            "train_augmentation" : train_augmentation
            }

def get_optimizer(optimizer_name, cosine_decay, learning_rate, num_epochs, momentum, decay):
    optimizer = None
    if optimizer_name == "SGD":
        if cosine_decay:
            schedule = CosineDecay(learning_rate, num_epochs)
            optimizer = SGD(learning_rate=schedule, momentum=momentum, decay=decay)
        else:
            optimizer = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay)

    elif optimizer_name == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = optimizer_name
        print("Taking {optimizer_name} as the optimizer")

    return optimizer
