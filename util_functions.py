import os
from tensorflow import keras
from MLP_mixer import MLPMixer


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
        "h14": {
            "num_blocks": 32,
            "patch_size": 14,
            "stem_width": 1280,
            "tokens_mlp_dim": 640,
            "channels_mlp_dim": 5120,
        },

    }

    return MODEL_CONFIGS[model_name]


def reload_model_weights(model, url=""):
    pretrained_dd = {
        "mlp_mixer_b16": ["imagenet", "imagenet21k"],
        "mlp_mixer_l16": ["imagenet", "imagenet21k"]
    }

    file_name = os.path.basename(url)

    try:
        pretrained_model = keras.utils.get_file(file_name, url, cache_subdir="models")
    except:
        print("[Error] will not load weights, url not found or download failed:", url)
        return
    else:
        print(">>>> Load pretraind from:", pretrained_model)
        model.load_weights(pretrained_model, by_name=True, skip_mismatch=True)


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
    if dataset.lower() == "flowers":
        return 102
    if dataset.lower() == "pets":
        return 37
    if dataset.lower() == "tiny-imagenet":
        return 200


# this function is only for retrieving one of the 4 pretrained models from the git repo or create a brand new untrained model
def get_model(configs):
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
        dropout=0,
        drop_connect_rate=0,
        initial_activation=configs["initial_activation"],
        mixer_activation="gelu",
        classifier_activation="softmax",
        model_name="mlp_mixer",
        pretrained=configs["pretrained"],
        local_model=configs["local_model_path"],
        url=configs["url"],
        unfreeze=configs["trainable_layers"]
    )
    return model


def generate_model_configs(pretrained="mlp_mixer_b16_imagenet",
                           dataset="cifar",
                           initial_activation="relu",
                           local_model_path="",
                           url="https://github.com/leondgarse/keras_mlp/releases/download/mlp_mixer/",
                           trainable_layers="top",
                           optimizer_name = "SGD",
                           learning_rate = 0.001,
                           num_epochs = 10,
                           momentum = 0.9,
                           decay = 0.001,
                           cosine_decay = False,
                           batch_size = 32
                           ):
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

    return {"pretrained": pretrained,
            "dataset_name": dataset,
            "initial_activation": initial_activation,
            "local_model_path": local_model_path,
            "url": url,
            "trainable_layers": trainable_layers,
            "optimizer": optimizer,
            "num_epochs": num_epochs,
            "batch_size": batch_size}

def get_generators(dataset):
    # Create an ImageDataGenerator with resizing
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(
        rescale=1. / 255
    )
    if dataset == "cifar10" or dataset == "tiny-imagenet":
        x_train_path, x_test_path = get_folder_path(dataset)
        train_generator = train_datagen.flow_from_directory(x_train_path, target_size=(224, 224), batch_size=32,
                                                            subset='training')
        validation_generator = train_datagen.flow_from_directory(x_train_path, target_size=(224, 224),
                                                                 batch_size=32,
                                                                 subset='validation')

        test_generator = test_datagen.flow_from_directory(x_test_path, target_size=(224, 224), batch_size=32)
    else:
        x_train_path, x_test_path = get_folder_path(dataset)
        x_train_df, x_test_df = get_dataframe(dataset)
        train_generator = train_datagen.flow_from_dataframe(x_train_path, target_size=(224, 224), batch_size=32,
                                                            subset='training')
        validation_generator = train_datagen.flow_from_dataframe(x_train_path, target_size=(224, 224),
                                                                 batch_size=32,
                                                                 subset='validation')

        test_generator = test_datagen.flow_from_dataframe(x_test_path, target_size=(224, 224), batch_size=32)

    return train_generator, validation_generator, test_generator

def get_folder_path(dataset):
    x_train_path = ""
    x_test_path = ""
    return x_train_path, x_test_path

def generate_data_directories():
    return