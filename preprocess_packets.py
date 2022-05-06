from common import *

def main(name_in, name_out):
    df = load_data(name_in)
    df['frame.time_epoch'] = df['frame.time_epoch'] * 1000
    df = df[['frame.time_epoch']]
    df.columns = ['press_time']
    df = combine_sessions(df, min_session_size=300, max_session_size=300, min_sessions=2, max_sessions=2)
    save_data(df, name_out, 'hdf')

if __name__ == '__main__':
    main('packets/aalto-eval-google_search.bak','packets/aalto-eval-google_search')
    main('packets/aalto-eval-ddg_search.bak','packets/aalto-eval-ddg_search')
