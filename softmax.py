from common import *


def RNNv0(input_shape, output_shape):
    inputs = layers.Input(shape=input_shape, dtype='float32', name='features')
    x = inputs
    x = layers.Masking(mask_value=PAD_VALUE)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(units=128, return_sequences=True)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(units=128, return_sequences=False)(x)
    x = layers.Dense(units=output_shape, activation='softmax')(x)

    model = models.Model(
    	inputs=[inputs],
    	outputs=x,
	    name='idnet')

    return model

def RNNv1(input_shape, output_shape):
    inputs = layers.Input(shape=input_shape, dtype='float32', name='features')
    x = layers.Masking(mask_value=PAD_VALUE)(inputs)
    x = layers.LSTM(units=64, return_sequences=False)(x)
    x = layers.Dense(units=64, activation='relu')(x)
    x = layers.Dense(units=output_shape, activation='softmax')(x)

    model = models.Model(
    	inputs=[inputs],
    	outputs=x,
	    name='idnet')

    return model


def RNNv2(input_shape, output_shape):
    inputs = layers.Input(shape=input_shape, dtype='float32', name='features')
    x = layers.Masking(mask_value=PAD_VALUE)(inputs)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dense(units=128, activation='relu')(x)
    x = layers.Dense(units=output_shape, activation='softmax')(x)

    model = models.Model(
    	inputs=[inputs],
    	outputs=x,
	    name='idnet')

    return model

def CNNv1(input_shape, output_shape):
    inputs = layers.Input(shape=input_shape, dtype='float32', name='features')
    x = layers.Masking(mask_value=PAD_VALUE)(inputs)
    x = tf.expand_dims(x, -1)
    x = layers.Conv2D(32, (3, 1), padding='valid', activation='relu')(x)
    x = layers.MaxPool2D((2,1), padding='same')(x)

    x = layers.Conv2D(64, (3, 1), padding='valid', activation='relu')(x)
    x = layers.MaxPool2D((2,1), padding='same')(x)

    x = layers.Conv2D(64, (3, 1), padding='valid', activation='relu')(x)
    x = layers.MaxPool2D((2,1), padding='same')(x)

    x = layers.Conv2D(64, (3, 1), padding='valid', activation='relu')(x)
    x = layers.MaxPool2D((2,1), padding='same')(x)

    x = layers.Conv2D(64, (3, 1), padding='valid', activation='relu')(x)
    x = layers.MaxPool2D((2,1), padding='same')(x)

    x = layers.Conv2D(64, (3, 1), padding='valid', activation='relu')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(units=output_shape, activation='softmax')(x)

    model = models.Model(
    	inputs=[inputs],
    	outputs=x,
	    name='idnet')

    return model


def train(name, dataset, model_fun, feature_fun, input_length, batch_size=1000, epochs=100, seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model_dir = os.path.join(MODELS_DIR, name)
    checkpoint_dir = os.path.join(model_dir, 'checkpoint')
    checkpoint_path = os.path.join(checkpoint_dir, 'model.hdf5')
    log_dir = os.path.join(model_dir, 'log')
    tensorboard_dir = os.path.join(log_dir, 'scalars')
    results_dir = os.path.join(model_dir, 'results')

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    cb_tensorboard = callbacks.TensorBoard(log_dir=tensorboard_dir)
    cb_history = callbacks.CSVLogger(os.path.join(log_dir, 'history.csv'), separator=',', append=False)
    cb_checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max', save_weights_only=False)

    df = load_data(dataset)
    df = feature_fun(df)

    # df = df.groupby(['user','session']).head(input_length)

    X, y = pad_and_split(df, input_length)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1/2, stratify=y)

    dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train), reshuffle_each_iteration=True).batch(batch_size)
    dataset_test = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    output_shape = len(np.unique(y))

    input_shape = X[0].shape
    model = model_fun(input_shape, output_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics='accuracy',
                  loss='sparse_categorical_crossentropy')

    print(model.summary())

    model.fit(dataset_train,
              validation_data=dataset_test,
              epochs=1000,
              callbacks=[cb_history,cb_tensorboard,cb_checkpoint],
              use_multiprocessing=True,
              verbose=1)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
      except RuntimeError as e:
        print(e)


    # train('RNNv0-softmax', 'packets/aalto-eval-google_search', RNNv0, features_tau, input_length=150)
    train('RNNv0-softmax-banerjee-user', 'keystrokes/banerjee-user', RNNv0, features_tau, input_length=1000)
    # train('RNNv0-softmax-banerjee-task', 'keystrokes/banerjee-task', RNNv0, features_tau, input_length=1000)
    # train('RNNv0-softmax-banerjee-true', 'keystrokes/banerjee-true', RNNv0, features_tau, input_length=1000)
