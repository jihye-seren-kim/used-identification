import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, pairwise_distances, average_precision_score
from sklearn.neighbors import KNeighborsClassifier

from common import *

OUTPUT_SHAPE = 64


def features_kd(df):
    idx = df.index.to_frame()
    session_start = (idx != idx.shift()).any(axis=1)

    t_down = df['press_time']/1000
    t_up = df['release_time']/1000

    f = pd.DataFrame(index=df.index)

    f['duration'] = t_up-t_down
    f['down_down'] = t_down.diff()
    f['up_up'] = t_up.diff()
    f['down_up'] = t_up-t_down.shift()
    f['up_down'] = t_down-t_up.shift()
    f = f.astype('float32')

    f['key_code'] = df['key_code']

    f.loc[session_start,['down_down','up_up','down_up','up_down']] = 0

    return f


def features_tau(df, max_tau=1000, min_tau=10):
    idx = df.index.to_frame()
    session_start = (idx != idx.shift()).any(axis=1)

    # millisecond timestamps
    t = df['press_time'].round()

    tau = t.diff()
    tau.loc[tau>max_tau] = max_tau
    tau.loc[tau<min_tau] = min_tau
    tau.loc[session_start] = PAD_VALUE

    f = pd.DataFrame(index=df.index)
    f['tau'] = tau
    f = f.astype('float32')
    return f


def features_tau2(df):
    idx = df.index.to_frame()
    session_start = (idx != idx.shift()).any(axis=1)

    t = df['press_time']/1000

    tau = t.diff()
    tau.loc[session_start] = 0

    tau2 = tau.diff()
    tau2.loc[session_start] = 0

    speed = 1*(tau2 > 0)
    speed.loc[tau2 < 0] = -1

    kps = 1/tau
    kps.loc[kps==np.inf] = np.NaN
    max_kps = kps.groupby(['user','session']).max()
    kps.loc[kps.isna()] = max_kps
    kps.loc[session_start] = 0

    f = pd.DataFrame(index=df.index)

    f['tau'] = tau
    # f['tau2'] = tau2
    # f['speed'] = speed
    # f['kps'] = kps

    f = f.astype('float32')

    # f = f.loc[~session_start,:]
    return f


def features_round(df):
    idx = df.index.to_frame()
    session_start = (idx != idx.shift()).any(axis=1)

    t = df['press_time']

    tau = t.diff()
    tau.loc[session_start] = 0
    tau = 1+np.ceil(tau/10)
    tau.loc[tau>200] = 200

    f = pd.DataFrame(index=df.index)
    f['tau'] = tau.astype('int32')
    f = f.loc[~session_start,:]
    return f


def features_round2(df, base=5, upper=2000):
    def custom_round(x, base):
        return int(base * round(float(x)/base))

    idx = df.index.to_frame()
    session_start = (idx != idx.shift()).any(axis=1)

    t = df['press_time']

    tau = t.diff()
    tau.loc[session_start] = 0
    tau = (tau/base).round()
    max_tau = upper/base
    tau.loc[tau>max_tau] = max_tau

    f = pd.DataFrame(index=df.index)
    f['tau'] = tau.astype('int32')
    # f = f.loc[~session_start,:]
    return f


def pad_and_split(df, length=None):
    idx, features = list(zip(*df.groupby(['user','session'])))
    y,_ = list(zip(*idx))
    y = np.array(y)
    y, _ = pd.factorize(y, sort=True)

    X = [f.values for f in features]
    # X = tf.keras.preprocessing.sequence.pad_sequences(X, dtype='float32', padding='post', maxlen=length, value=PAD_VALUE)
    X = np.array(X)

    return X, y


def model_accuracy(X, y, num_training_samples=1, seed=SEED):
    X_train, X_test, y_train, y_test = train_test_split_balanced(X, y, train_size=1, random_state=seed)
    clf = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    return acc


class Checkpoint(callbacks.Callback):
    def __init__(self, ckpt, manager, model, X, y, eval_period=None, model_path=None):
        self.ckpt = ckpt
        self.manager = manager
        self.model = model
        self.X = X
        self.y = y
        self.eval_period = eval_period
        self.model_path = model_path

    def on_epoch_end(self, epoch, logs={}):
        self.ckpt.step.assign_add(1)
        print()

        if (epoch % self.eval_period) == 0:
            X_embed = self.model.predict(self.X)

            acc = model_accuracy(X_embed, self.y)
            num_classes = len(np.unique(self.y))

            print(f'Epoch {int(self.ckpt.step)}: {acc:.4f} ACC (previous best is {float(self.ckpt.val_acc):.4f} with {num_classes} classes)')
            tf.summary.scalar('validation_acc', data=acc, step=epoch)

            if acc >= float(self.ckpt.val_acc):
               self.ckpt.val_acc.assign(acc)
               if self.model_path is not None:
                   self.model.save(self.model_path)
                   print(f'******** Saved new best model with validation ACC of {float(self.ckpt.val_acc):.4f} ********')

        save_path = self.manager.save()
        print('Saved checkpoint for epoch %d (%.4f VAL ACC): %s' % (int(self.ckpt.step), float(self.ckpt.val_acc), save_path))


def TypeNet(input_shape):
    inputs = layers.Input(shape=input_shape, dtype='float32', name='features')
    x = inputs
    x = layers.Masking(mask_value=PAD_VALUE)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(units=128, return_sequences=True)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(units=128, return_sequences=False)(x)
    x = layers.Lambda(lambda x: K.l2_normalize(x, axis=-1))(x)

    model = models.Model(
    	inputs=[inputs],
    	outputs=x,
	    name='idnet')

    return model


def PacketNet(input_shape):
    inputs = layers.Input(shape=input_shape, dtype='float32', name='features')
    x = inputs[:,:,0]
    x = layers.Embedding(input_dim=201, output_dim=16, mask_zero=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(units=128, return_sequences=True)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(units=128, return_sequences=False)(x)
    x = layers.Lambda(lambda x: K.l2_normalize(x, axis=-1))(x)

    model = models.Model(
    	inputs=[inputs],
    	outputs=x,
	    name='idnet')

    return model


def PacketNet2(input_shape):
    inputs = layers.Input(shape=input_shape, dtype='float32', name='features')
    x = inputs[:,:,0]
    x = layers.Embedding(input_dim=401, output_dim=16, mask_zero=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(units=128, return_sequences=True)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(units=128, return_sequences=False)(x)
    x = layers.Lambda(lambda x: K.l2_normalize(x, axis=-1))(x)

    model = models.Model(
    	inputs=[inputs],
    	outputs=x,
	    name='idnet')

    return model


def RNNv1(input_shape):
    inputs = layers.Input(shape=input_shape, dtype='float32', name='features')
    x = inputs
    x = layers.Masking(mask_value=PAD_VALUE)(x)
    x = layers.LSTM(units=OUTPUT_SHAPE, return_sequences=False)(x)
    x = layers.Lambda(lambda x: K.l2_normalize(x, axis=-1))(x)

    model = models.Model(
    	inputs=[inputs],
    	outputs=x,
	    name='idnet')

    return model


def RNNv2(input_shape):
    inputs = layers.Input(shape=input_shape, dtype='float32', name='features')
    x = inputs
    # x = layers.Masking(mask_value=PAD_VALUE)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dense(units=64, activation='relu')(x)
    x = layers.Dense(units=OUTPUT_SHAPE, activation=None)(x)
    x = layers.Lambda(lambda x: K.l2_normalize(x, axis=-1))(x)

    model = models.Model(
    	inputs=[inputs],
    	outputs=x,
	    name='idnet')

    return model


def RNNv3(input_shape):
    inputs = layers.Input(shape=input_shape, dtype='float32', name='features')
    x = inputs
    # x = layers.Masking(mask_value=PAD_VALUE)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dense(units=OUTPUT_SHAPE, activation=None)(x)
    x = layers.Lambda(lambda x: K.l2_normalize(x, axis=-1))(x)

    model = models.Model(
    	inputs=[inputs],
    	outputs=x,
	    name='idnet')

    return model


def RNNv4(input_shape):
    inputs = layers.Input(shape=input_shape, dtype='float32', name='features')
    x = inputs[:,:,0]
    x = layers.Embedding(input_dim=201, output_dim=16, mask_zero=True)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dense(units=OUTPUT_SHAPE, activation=None)(x)
    x = layers.Lambda(lambda x: K.l2_normalize(x, axis=-1))(x)

    model = models.Model(
    	inputs=[inputs],
    	outputs=x,
	    name='idnet')

    return model


def RNNv5(input_shape):
    inputs = layers.Input(shape=input_shape, dtype='float32', name='features')
    x = inputs[:,:,0]
    x = layers.Embedding(input_dim=201, output_dim=16, mask_zero=False)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dense(units=OUTPUT_SHAPE, activation=None)(x)
    x = layers.Lambda(lambda x: K.l2_normalize(x, axis=-1))(x)

    model = models.Model(
    	inputs=[inputs],
    	outputs=x,
	    name='idnet')

    return model


def train(name, data_train, data_eval, model_fun, feature_fun, input_length, seed=SEED, batch_size=256, epochs=1000, eval_period=1, train_users=None, eval_users=None):
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

    df_train = load_data(data_train)
    df_eval = load_data(data_eval)

    if train_users is not None:
        # train_idx = np.random.choice(df_train.index.unique('user'), train_users, replace=False)
        train_idx = df_train.index.unique('user')[:train_users]
        df_train = df_train.loc[df_train.index.get_level_values('user').isin(set(train_idx)),:]

    if eval_users is not None:
        # eval_idx = np.random.choice(df_eval.index.unique('user'), eval_users, replace=False)
        eval_idx = df_eval.index.unique('user')[:eval_users]
        df_eval = df_eval.loc[df_eval.index.get_level_values('user').isin(set(eval_idx)),:]

    df_train = feature_fun(df_train)
    df_eval = feature_fun(df_eval)

    # df_train = df_train.groupby(['user','session']).head(input_length)
    # df_eval = df_eval.groupby(['user','session']).head(input_length)

    X_train, y_train = pad_and_split(df_train, input_length)
    X_eval, y_eval = pad_and_split(df_eval, input_length)

    X_eval = tf.data.Dataset.from_tensor_slices(X_eval)
    X_eval = X_eval.batch(min(batch_size, len(X_eval)))

    num_samples = len(y_train)
    num_users = len(np.unique(y_train))

    split_idx = np.unique(y_train, return_index=True)[1][1:]
    X_train = np.split(X_train, split_idx)
    y_train = np.split(y_train, split_idx)

    dataset = tf.data.Dataset.from_generator(lambda: zip(X_train,y_train), (tf.float32, tf.int64))
    dataset = dataset.shuffle(num_users, reshuffle_each_iteration=True, seed=dataset_seed)
    dataset = dataset.unbatch()
    dataset = dataset.batch(min(batch_size, num_samples), drop_remainder=True)

    input_shape = (None, X_train[0].shape[-1])
    model = model_fun(input_shape)

    optimizer = tf.keras.optimizers.Adam(0.001)
    model.compile(optimizer=optimizer,
                  loss=tfa.losses.TripletSemiHardLoss())
                  # loss=tfa.losses.TripletSemiHardLoss(margin=2.0))
                  # loss=tfa.losses.TripletHardLoss(margin=1.5))
                  # loss=tfa.losses.TripletHardLoss(margin=0.2))

    ckpt = tf.train.Checkpoint(step=tf.Variable(0), val_acc=tf.Variable(0.), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

    print(model.summary())
    print('Train users:', num_users)
    print('Eval users :', len(np.unique(y_eval)))
    print('Dataset seed', dataset_seed)

    cb_checkpoint = Checkpoint(ckpt, manager, model, X_eval, y_eval, eval_period=eval_period, model_path=model_path)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    model.fit(dataset,
              initial_epoch=int(ckpt.step),
              epochs=epochs,
              callbacks=[cb_history,cb_tensorboard,cb_checkpoint],
              use_multiprocessing=True,
              workers=16,
              max_queue_size=32,
              verbose=1)


def test(name, dataset, output, feature_fun, batch_size=1000, keep_features=None, num_events=None, num_users=None, keep_rate=None, shuffle=False, top_k=[1,5,10,50,100]):
    df_eval = load_data(dataset)

    if num_users is not None:
        np.random.seed(SEED)
        # train_idx = np.random.choice(df_train.index.unique('user'), train_users, replace=False)
        user_idx = df_eval.index.unique('user')[:num_users]
        df_eval = df_eval.loc[df_eval.index.get_level_values('user').isin(set(user_idx)),:]

    if num_events is not None:
        df_eval = df_eval.groupby(['user','session']).head(num_events)

    if keep_rate is not None:
        df_eval = df_eval.groupby(['user','session']).sample(frac=keep_rate, random_state=SEED)
        df_eval = df_eval.groupby(['user','session']).apply(lambda x: x.sort_values('press_time'))

    df_eval = feature_fun(df_eval)

    if shuffle:
        df_eval = df_eval.groupby(['user','session']).sample(frac=1, random_state=SEED)

    X_eval, y_eval = pad_and_split(df_eval)

    num_users = len(np.unique(y_eval))

    X_eval = tf.data.Dataset.from_tensor_slices(X_eval)
    X_eval = X_eval.batch(batch_size)

    model_path = os.path.join(MODELS_DIR, name, 'model', 'model.hdf5')
    model = models.load_model(model_path, compile=False)

    X_eval_embeddings = model.predict(X_eval)

    X_train, X_test, y_train, y_test = train_test_split_balanced(X_eval_embeddings, y_eval, train_size=1, random_state=SEED)
    dists = pairwise_distances(X_train, X_test, n_jobs=64)
    dists = pd.DataFrame(dists, index=y_train, columns=y_test).sort_index(axis=0).sort_index(axis=1)

    # Average distances to each training sample if there are multiple
    dists = dists.groupby(level=0).mean()

    # Average distances between train and test samples
    # dists = dists.groupby(level=0, axis=0).mean().groupby(level=0, axis=1).mean()

    ranks = dists.apply(lambda x: x.argsort().argsort().loc[x.name])

    m = {}
    m['num_users'] = len(np.unique(y_eval))
    m['acc'] = (dists.idxmin() == dists.columns).sum()/len(dists.columns)

    for k in top_k:
        m['rank_%d' % k] = (ranks < k).sum()/len(ranks)

    same = dists.index.values[:,np.newaxis] == dists.columns.values

    from sklearn.metrics import roc_curve
    fpr, tpr, thresh = roc_curve(same.flatten(), -dists.values.flatten())
    fnr = 1 - tpr

    eer_idx = np.abs(fpr-fnr).argmin()
    eer = np.mean([fpr[eer_idx], fnr[eer_idx]])
    tpr1e3 = tpr[np.abs(fpr - 1e-3).argmin()]

    m['eer'] = eer
    m['tpr1e3'] = tpr1e3

    m = pd.Series(m)
    print(m)
    save_results(m, output)

    # features = pd.DataFrame(X_eval_embeddings, index=pd.Index(y_eval, name='user'))
    # features.to_hdf(os.path.join(FEATURES_DIR, out_name), key='key')


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
      except RuntimeError as e:
        print(e)

    # train('typenet', 'keystrokes/aalto-train', 'keystrokes/aalto-eval', TypeNet, features_kd, input_length=300)
    test('typenet', 'keystrokes/aalto-eval', 'aalto-eval-kd', features_kd, num_events=300, num_users=500)

    # train('packetnet', 'keystrokes/aalto-train', 'keystrokes/aalto-eval', PacketNet, features_round2, input_length=300)#, train_users=10000)
    # test('packetnet', 'keystrokes/aalto-eval', 'aalto-eval-tau', features_round2, num_events=300, num_users=500)
    # test('packetnet', 'packets/aalto-eval-google_search', 'aalto-eval-packets-tau', features_round2, num_events=300, num_users=500)

    # train('packetnet2', 'keystrokes/aalto-train', 'keystrokes/aalto-eval', PacketNet2, features_round2, input_length=300)#, train_users=10000)
    test('packetnet2', 'keystrokes/aalto-eval', 'aalto-eval-tau', features_round2, num_events=300, num_users=500)
    test('packetnet2', 'packets/aalto-eval-google_search', 'aalto-eval-packets-tau', features_round2, num_events=300, num_users=500)

    for i in range(100, 501, 10):
        test('packetnet2', 'keystrokes/aalto-eval', 'aalto-eval-tau-%s_users' % i, features_round2, num_events=300, num_users=i)
        test('packetnet2', 'packets/aalto-eval-google_search', 'aalto-eval-packets-tau-%s_users' % i, features_round2, num_events=300, num_users=i)

    for i in range(50, 301, 5):
        test('packetnet2', 'keystrokes/aalto-eval', 'aalto-eval-tau-%s_keystrokes' % i, features_round2, num_events=i, num_users=500)
        test('packetnet2', 'packets/aalto-eval-google_search', 'aalto-eval-packets-tau-%s_keystrokes' % i, features_round2, num_events=i, num_users=500)

    for i in range(5,101,5):
        test('packetnet2', 'keystrokes/aalto-eval', 'aalto-eval-tau-%s_detection' % i, features_round2, keep_rate=i/100, num_users=500)
        test('packetnet2', 'packets/aalto-eval-google_search', 'aalto-eval-packets-tau-%s_detection' % i, features_round2, keep_rate=i/100, num_users=500)
