from common import *


class PrintPreds(callbacks.Callback):
    def __init__(self, model, X, y, eval_period):
        self.model = model
        self.X = X
        self.y = y
        self.eval_period = eval_period

    def on_epoch_end(self, epoch, logs={}):
        np.set_printoptions(precision=3)
        if (epoch % self.eval_period) == 0:
            print()
            idx = np.random.choice(np.arange(len(self.X)))
            y_pred = self.model.predict(self.X[[idx]])[0]
            y_true = self.y[idx]

            print('Pred:', y_true[:10].squeeze())
            print('True:', y_pred[:10].squeeze())


def Seqv0(input_shape):
    inputs = layers.Input(shape=input_shape, dtype='float32', name='features')
    x = inputs
    x = layers.Masking(mask_value=PAD_VALUE)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dense(1, activation=None)(x)

    model = models.Model(
    	inputs=[inputs],
    	outputs=x,
	    name='idnet')

    return model


def Seqv1(input_shape):
    inputs = layers.Input(shape=input_shape, dtype='float32', name='features')
    x = inputs
    x = layers.Masking(mask_value=PAD_VALUE)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dense(1, activation=None)(x)

    model = models.Model(
    	inputs=[inputs],
    	outputs=x,
	    name='idnet')

    return model


def make_samples(df, length):
    idx = df.index.to_frame()
    session_start = (idx != idx.shift()).any(axis=1)

    t_press = df['press_time']/1000
    t_release = df['release_time']/1000

    tau = t_press.diff()
    tau.loc[session_start] = 0

    duration = t_release - t_press

    tmp = pd.concat([tau.rename('tau'), duration.rename('duration')], axis=1)
    idx_unique, dfs = zip(*tmp.groupby(['user','session']))
    idx_unique = list(idx_unique)

    X = [f[['tau']].values for f in dfs]
    X = tf.keras.preprocessing.sequence.pad_sequences(X, dtype='float32', padding='post', maxlen=length, value=PAD_VALUE)

    y = [f[['duration']].values for f in dfs]
    y = tf.keras.preprocessing.sequence.pad_sequences(y, dtype='float32', padding='post', maxlen=length, value=PAD_VALUE)

    return X, y, idx_unique


def make_samples_test(df, length):
    idx = df.index.to_frame()
    session_start = (idx != idx.shift()).any(axis=1)

    t_press = df['press_time']/1000

    tau = t_press.diff()
    tau.loc[session_start] = 0

    idx_unique, dfs = zip(*tau.groupby(['user','session']))
    idx_unique = list(idx_unique)

    X = [f.values for f in dfs]
    X = tf.keras.preprocessing.sequence.pad_sequences(X, dtype='float32', padding='post', maxlen=length, value=PAD_VALUE)

    return X, idx_unique


def train(name, data_train, data_eval, model_fun, input_length, seed=SEED, batch_size=300, epochs=1000, eval_period=1, train_users=None, eval_users=None):
    dataset_seed = np.random.randint(1000,1000000) # note, this is before any global seeds are set
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model_dir = os.path.join(MODELS_DIR, name)

    # path to the best saved model
    save_model_dir = os.path.join(model_dir, 'model')
    model_path = os.path.join(save_model_dir, 'model.hdf5')

    log_dir = os.path.join(model_dir, 'log')
    tensorboard_dir = os.path.join(log_dir, 'scalars')
    results_dir = os.path.join(model_dir, 'results')

    # path to checkpoint to resume training
    checkpoint_dir = os.path.join(model_dir, 'checkpoint')

    file_writer = tf.summary.create_file_writer(os.path.join(tensorboard_dir, 'metrics'))
    file_writer.set_as_default()

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    cb_tensorboard = callbacks.TensorBoard(log_dir=tensorboard_dir)
    cb_history = callbacks.CSVLogger(os.path.join(log_dir, 'history.csv'), separator=',', append=False)

    df_train = load_data(data_train).sort_index()
    df_eval = load_data(data_eval).sort_index()

    if train_users is not None:
        # train_idx = np.random.choice(df_train.index.unique('user'), train_users, replace=False)
        train_idx = df_train.index.unique('user')[:train_users]
        df_train = df_train.loc[df_train.index.get_level_values('user').isin(set(train_idx)),:]

    if eval_users is not None:
        eval_idx = np.random.choice(df_eval.index.unique('user'), eval_users, replace=False)
        eval_idx = df_eval.index.unique('user')[:eval_users]
        df_eval = df_eval.loc[df_eval.index.get_level_values('user').isin(set(eval_idx)),:]

    df_train = df_train.groupby(['user','session']).head(input_length)
    df_eval = df_eval.groupby(['user','session']).head(input_length)

    X_train, y_train, idx_train = make_samples(df_train, input_length)
    X_eval, y_eval, idx_eval = make_samples(df_eval, input_length)

    dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset_train = dataset_train.shuffle(1500)
    dataset_train = dataset_train.batch(batch_size, drop_remainder=True)

    dataset_eval = tf.data.Dataset.from_tensor_slices((X_eval, y_eval))
    dataset_eval = dataset_eval.batch(batch_size)

    input_shape = X_train[0].shape
    model = model_fun(input_shape)

    optimizer = tf.keras.optimizers.Adam(0.001)
    model.compile(optimizer=optimizer,
                  loss='mse')

    ckpt = tf.train.Checkpoint(step=tf.Variable(0), val_acc=tf.Variable(0.), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

    print(model.summary())

    cb_checkpoint = callbacks.ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss', mode='min', save_weights_only=False, verbose=1)

    cb_print = PrintPreds(model, X_eval, y_eval, eval_period=1)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    model.fit(dataset_train,
              validation_data=dataset_eval,
              initial_epoch=int(ckpt.step),
              epochs=epochs,
              callbacks=[cb_history,cb_tensorboard,cb_checkpoint,cb_print],
              use_multiprocessing=True,
              workers=16,
              max_queue_size=32,
              verbose=1)


def test(name, dataset, output, input_length, batch_size=300):
    df_eval = load_data(dataset)[['press_time']]
    df_eval = df_eval.groupby(['user','session']).head(input_length)

    X_eval, idx = make_samples_test(df_eval, input_length)
    df_eval = df_eval.loc[idx]

    dataset_eval = tf.data.Dataset.from_tensor_slices(X_eval)
    dataset_eval = dataset_eval.batch(batch_size, drop_remainder=True)

    model_path = os.path.join(MODELS_DIR, name, 'model', 'model.hdf5')
    model = models.load_model(model_path, compile=False)

    y_pred = model.predict(X_eval).squeeze()

    lens = df_eval.groupby(['user','session']).size().loc[idx].values
    y_pred = np.concatenate([y[:n] for y,n in zip(y_pred,lens)])

    duration = (1000*y_pred).round().astype(df_eval.dtypes['press_time'])
    df_eval['release_time'] = df_eval['press_time'] + duration

    save_data(df_eval, output, ext='hdf')


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
      except RuntimeError as e:
        print(e)

    # train('Seqv1', 'keystrokes/aalto-train', 'keystrokes/aalto-eval', Seqv1, input_length=150)
    test('Seqv1', 'keystrokes/aalto-eval', 'predictions/aalto-eval', input_length=150)
