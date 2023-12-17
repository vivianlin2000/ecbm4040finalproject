from model_utils import *
import os
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from data_utils import *
import matplotlib.pyplot as plt


def set_seed(seed=31415):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seed()

def train_model(model, configs):
    

    
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
    
    if pretrained:
        pretrained_name = pretrained[pretrained.rfind('_', 0, pretrained.rfind('_'))+1:]
        expt_name = f"{pretrained_name}_{dataset_name}"
        
    else:
        expt_name = f"{model_name}_{dataset_name}"
        
    saved_models_dir = f"saved_models/{expt_name}"
    os.makedirs(f"{saved_models_dir}", exist_ok = True)
    filename = f"{expt_name}_epochs_{num_epochs}_{optimizer_name}_{learning_rate}"

    train_generator, validation_generator, test_generator = get_generators(dataset_name, batch_size, pretrained, train_augmentation)
    
    
    optimizer = get_optimizer(optimizer_name, cosine_decay, learning_rate, num_epochs, momentum, decay)
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])


    callbacks = get_callbacks(saved_models_dir, filename)
    
    print_model_trainable_params(model)
    print('\nBeginning Training:\n')

    history = model.fit(x=train_generator, max_queue_size=10, workers=16, batch_size=batch_size, epochs=num_epochs,
                        validation_data=validation_generator, callbacks=callbacks)
    
    

    # Call the function with the training history
    plot_and_save_history(history,filename=f"{saved_models_dir}/PLOTS_{filename}" )
    
    # Call the function with the training history
    save_history_to_csv(history,filename=f"{saved_models_dir}/history_{filename}.csv")
    
    print(f"\nTraining Finished. \nYou can find Saved model (best validation score), TensorBoard logs, Loss curves and Accuracy curves, csv of History,  at {saved_models_dir}")
    
    print("Now loading Best model and Testing..")
    test(saved_models_dir, filename, test_generator)


def print_model_trainable_params(model):
    trainable_count = np.sum([K.count_params(w) \
                              for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) \
                                  for w in model.non_trainable_weights])
    

    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))


def get_callbacks(saved_models_dir, filename):
    model_checkpoint = ModelCheckpoint(
        filepath=f"{saved_models_dir}/{filename}.h5",
        monitor='val_acc',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    log_dir = f"{saved_models_dir}/TBlogs"
    os.makedirs(f"{log_dir}", exist_ok=True)

    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)
    callbacks = [model_checkpoint, tensorboard_callback, early_stopping_callback]

    return callbacks

def plot_and_save_history(history, filename='training_history.png'):
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
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(filename, index=False)

def test(saved_models_dir, filename, test_generator):
    file_path = f"{saved_models_dir}/{filename}.h5"
    model = load_model(file_path)

    test_results = model.evaluate(test_generator)

    print("Test Loss:", test_results[0])
    print("Test Accuracy:", test_results[1])

    with open(f"{saved_models_dir}/test_accuracy.txt", 'w') as file:
        file.write(f'Test Accuracy: {test_results[1]}\n')
    