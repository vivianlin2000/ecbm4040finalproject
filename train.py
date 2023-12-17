from util_functions import *

def train_model(model, configs):
    # Reproducability
    def set_seed(seed=31415):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

    set_seed()

    num_epochs = configs["num_epochs"]
    batch_size = configs["batch_size"]

    expt_name = "CIFAR_C_2_FT_B16_IM"
    saved_models_dir = f"saved_models/{configs['dataset_name']}/{expt_name}"
    os.makedirs(f"{saved_models_dir}", exist_ok = True)
    filename = f"{expt_name}_epochs_{num_epochs}_Adam_{learning_rate}_dropout_0.1"

    train_generator, validation_generator, test_generator = get_generators(configs["dataset_name"])
    # Compile the model
    model.compile(optimizer=configs["optimizer"], loss='categorical_crossentropy', metrics=['acc'])

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

    trainable_count = np.sum([K.count_params(w) \
                              for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) \
                                  for w in model.non_trainable_weights])
    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))
    print('\nBeginning Training:\n')

    history = model.fit(x=train_generator, max_queue_size=10, workers=16, batch_size=batch_size, epochs=num_epochs,
                        validation_data=validation_generator, callbacks=callbacks)

    return history