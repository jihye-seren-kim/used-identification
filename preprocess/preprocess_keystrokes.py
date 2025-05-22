from common import *
import string
from keycodes import KEEP_CODES

def main(name, num_eval_users=1000, min_events=300, max_events=300, min_sessions=2, max_sessions=2, seed=SEED):
    df = load_data(name)#, start=0, stop=1000000)
    
    # drop anything except alphanumeric and punctuation
    df = df.loc[df['key_code'].isin(KEEP_CODES)]

    # merge sessions
    df = combine_sessions(df, min_session_size=min_events, max_session_size=max_events, min_sessions=min_sessions, max_sessions=max_sessions, sort_col='press_time')

    # split dataset
    users = df.index.unique('user')

    np.random.seed(seed)
    users = np.random.permutation(users)
    eval_users = users[:num_eval_users]
    train_users = users[num_eval_users:]
    save_data(df.loc[train_users].sort_index(), name + '-train', 'hdf')
    save_data(df.loc[eval_users].sort_index(), name + '-eval', 'hdf')


if __name__ == '__main__':
    main('keystrokes/aalto', num_eval_users=1000)
