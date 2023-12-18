from model_utils import *
import os
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from data_utils import *
import matplotlib.pyplot as plt
from time import time as T


def set_seed(seed=2121):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): Seed value for random number generation.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seed()

def train_model(model, configs):
    """
    Train a MLP Mixer model based on the provided configurations.

    Args:
        model (tf.keras.Model): The MLP Mixer model instance.
        configs (dict): Dictionary containing model configurations.

    Returns:
        model (tf.keras.Model): The model with the highest validation accuracy.
        saved_models_dir (str): The directory path where the model, logs, curves, history, and time are saved.
        test_generator (tf.keras.utils.Sequence): The data generator for the test set, used for evaluating the model.
    """
    
    # Extract configurations
    num_epochs = configs["num_epochs"]
    batch_size = configs["batch_size"]
    learning_rate = configs["learning_rate"]
    model_name = configs["model_name"]
    optimizer_name = configs["optimizer_name"]
    dataset_name = configs['dataset_name']
    cosine_decay = configs["cosine_decay"]
    momentum = configs["momentum"]
    decay = configs["decay"]
    train_augmentation = configs["train_augmentation"]
    pretrained = configs['pretrained']
    
    # Generate experiment name based on whether the model is pretrained or not
    if pretrained:
        pretrained_name = pretrained[pretrained.rfind('_', 0, pretrained.rfind('_'))+1:]
        expt_name = f"{pretrained_name}_{dataset_name}"
        
    else:
        expt_name = f"{model_name}_{dataset_name}"
    
    # Create directory to save models
    saved_models_dir = f"saved_models/{expt_name}"
    os.makedirs(f"{saved_models_dir}", exist_ok = True)
    filename = f"{expt_name}_epochs_{num_epochs}_{optimizer_name}_{learning_rate}"
    
    # Get data generators
    train_generator, validation_generator, test_generator = get_generators(dataset_name, batch_size, pretrained, train_augmentation)
    
    # Get optimizer and compile the model
    optimizer = get_optimizer(optimizer_name, cosine_decay, learning_rate, num_epochs, momentum, decay)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

    # Get callbacks for training
    callbacks = get_callbacks(saved_models_dir, filename)
    
    print_model_trainable_params(model)
    
    print('\nBeginning Training:\n')
    # Start recording the time
    start_time = T()
    
    # The main training function. Takes the generators, queue and workers, batch size, epochs and callbacks.
    history = model.fit(x=train_generator, max_queue_size=10, workers=16, batch_size=batch_size, epochs=num_epochs,
                        validation_data=validation_generator, callbacks=callbacks)
    
    # Record end of training time
    end_time = T()
    elapsed_time = end_time - start_time
    
    # Plot the training history
    plot_and_save_history(history,filename=f"{saved_models_dir}/PLOTS_{filename}" )
    
    # Save training history
    save_history_to_csv(history,filename=f"{saved_models_dir}/history_{filename}.csv")
    
    # Save the training time
    print(f"\nTraining Finished. Time taken : {elapsed_time} seconds")
    with open(f"{saved_models_dir}/time_taken.txt", 'w') as file:
        file.write(f'Time taken : {elapsed_time} seconds \n')
          
    print(f"\nYou can find Saved model (best validation score), TensorBoard logs, Loss curves and Accuracy curves, csv of History,  at {saved_models_dir}")
    
    # Retrieve the model saved in the directory (since it is the best model with highest val accuracy)
    file_path = f"{saved_models_dir}/{filename}.h5"
    model = load_model(file_path)
    
    print("\nReturning the Best model, saved models directory, and test generator")
    return model, saved_models_dir, test_generator



def print_model_trainable_params(model):
    """
    Print the total, trainable, and non-trainable parameters of a Keras model.

    Args:
        model (tf.keras.Model): Keras model instance.

    """
    
    # Just get list of parameters from the model and sum
    trainable_count = np.sum([K.count_params(w) \
                              for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) \
                                  for w in model.non_trainable_weights])
    

    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))


def get_callbacks(saved_models_dir, filename):
    """
    Get a list of callbacks for model training.

    Args:
        saved_models_dir (str): Directory to save models.
        filename (str): Model filename.

    Returns:
        list: List of Keras callbacks.
    """
    # ModelCheckpoint callback: Saves the model with the best validation accuracy
    model_checkpoint = ModelCheckpoint(
        filepath=f"{saved_models_dir}/{filename}.h5",
        monitor='val_acc',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # TensorBoard callback: Logs training information for visualization in TensorBoard
    log_dir = f"{saved_models_dir}/TBlogs"
    os.makedirs(f"{log_dir}", exist_ok=True)
    
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # EarlyStopping callback: Stops training when a monitored quantity has stopped improving after 5 epochs
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)
    
    # Final List of callbacks to be used during training
    callbacks = [model_checkpoint, tensorboard_callback, early_stopping_callback]

    return callbacks

def plot_and_save_history(history, filename='training_history.png'):
    """
    Plot and save training and validation accuracy and loss curves.

    Args:
        history (tf.keras.callbacks.History): Training history object.
        filename (str): Name of the file to save the plots.

    """
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig( f"{filename}_accuracy.png")
    plt.show()

    plt.clf()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig( f"{filename}_loss.png" )
    plt.show()
        
def save_history_to_csv(history, filename='training_history.csv'):
    """
    Save training history to a CSV file.

    Args:
        history (tf.keras.callbacks.History): Training history object.
        filename (str): Name of the CSV file.
    """
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(filename, index=False)


    