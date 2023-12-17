from model_utils import get_model, generate_model_configs
from train import train_model


User_Configs = {
    # Choose pretrained as None, "mlp_mixer_b16_imagenet", "mlp_mixer_l16_imagenet", "mlp_mixer_b16_imagenet21k", or "mlp_mixer_l16_imagenet21k"
    'pretrained': None,
    
    # choose model name as 
    # (a). One of "b16" / "l16" corresponding to pretrained model, or 
    # (b). One of b16/l16/s16/b32/l32/s32 if training from Scratch.
    'model_name': 'l32',
    
    # dataset_name should be one of "cifar10", "pets", or tiny-imagenet"
    'dataset_name': 'tiny-imagenet',
    
    # You can choose to not have activation in the first layer (linear projection), or else relu works well.
    'initial_activation': 'relu',
    
    # If you wish to reload a trained model for further training or testing, supply absolute local path in full.
    'local_model_path': None,
    
    # Set Learning Rate. 0.001-0.01 works best. Set low if finetuning.
    'learning_rate': 0.001,
    
    # (Relevant if Pretrained) Select only Top layer to be unfrozen by saying "top", or None for all unfrozen. 
    'trainable_layers': 'top',
    
    # Number of epochs to run training. About 10-20 epochs take roughly 1-3 hours on most models.
    'num_epochs': 30,
    
    # Optimizer to use. select "Adam" or "SGD", or any other.
    'optimizer_name': 'Adam',
    
    # (Relevant if SGD) Momentum and decay if using SGD
    'momentum': 0.9,
    'decay': 0.001,
    
    # Select Batch Size. 32 works well.
    'batch_size': 64,
    
    # Dropout for last layer and DropConnect for Mixer layers.
    'dropout': 0,
    'drop_connect_rate': 0,
    
    # Augmenting images during training. A combination of flipping, rotation, shears, zooms will be applied.
    "train_augmentation" : True
}



configs = generate_model_configs()
configs.update(User_Configs)
print("Here are the Final configurations : \n")
for key, value in configs.items():
    print(key, "=", value)
    
    
model = get_model(configs)

train_model(model, configs)