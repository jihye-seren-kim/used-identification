import os
import sys
import urllib.parse as urlparse
from io import StringIO
from subprocess import check_output
from glob import glob

from common import *

SKIP_FINISHED = True
MOVE_FAILURES = True

#172.217.5.100
CLIENT_IP = '10.0.0.182'

TSHARK_HTTP_CSV = 'tshark -r %s -o ssl.keylog_file:%s' \
' -T fields -e frame.number -e ip.src -e ip.dst -e frame.time -e tcp.len -e _ws.col.Protocol' \
' -e http.request.method -e http.request.uri' \
' -e http2.headers.method -e http2.headers.path' \
' -E header=y -E separator=, -E quote=d'

# TSHARK_HTTP_CSV = 'tshark -r %s -o ssl.keylog_file:%s' \
# ' -T fields -e frame.number -e ip.src -e ip.dst -e frame.time -e ssl.record.length -e _ws.col.Protocol' \
# ' -e http.request.method -e http.request.uri' \
# ' -e http2.headers.method -e http2.headers.path' \
# ' -E header=y -E separator=, -E quote=d'


MAX_KEYS_PER_QUERY = 3
MAX_MEAN_KEYS_PER_QUERY = 1.5
MAX_RATIO_MULTI_KEYS = 0.5

def detection_truth(df, website):
    if website == 'google':
        ac_path = '/complete/search?'
        ac_key = 'q'
    elif website == 'bing':
        ac_path = '/AS/Suggestions?'
        ac_key = 'qry'
    elif website == 'duckduckgo':
        ac_path = 'callback=autocompleteCallback'
        ac_key = 'q'
    elif website == 'yandex':
        ac_path = '/suggest-spok/suggest-ya.cgi'
        ac_key = 'part'
    elif website == 'baidu':
        ac_path = '/su?'
        # ac_path = '/s?'
        ac_key = 'wd'

    df = df[df['path'].str.contains(ac_path, regex=False).fillna(False)].copy()
    df['query'] = df['path'].apply(lambda path: urlparse.parse_qs(urlparse.urlparse(path).query)[ac_key][0])
    df['query_len'] = df['query'].str.len()
    df = df.sort_values('query_len')
    df['keys'] = [q[l:] for q,l in zip(df['query'].values, df['query'].str.len().shift().fillna(0).astype(int))]
    df = df.sort_values('frame_time')

    return df


def convert_pcap(name):
    fname_keys, fname_pcap, fname_ssl = capture_raw_fnames(name)
    fname_pre, fname_post, fname_truth = capture_fnames(name)

    (capture_type, browser, website, trial) = name.split('_')

    # Process pcap
    try:
        out = check_output((TSHARK_HTTP_CSV % (fname_pcap, fname_ssl)).split(' ')).decode('utf-8')
    except Exception:
        print('Running tshark failed, skipping', name, flush=True)
        return False

    post = pd.read_csv(StringIO(out), index_col=0, parse_dates=['frame.time']).sort_values('frame.time')
    post.index.name = 'idx'
    post.columns = ['src','dst','frame_time','frame_length','protocol',
                    'http2_method','http2_path', 'http_method','http_path']
    post['frame_time'] = post['frame_time'].astype(int).values/1e6
    # Only outbound traffic
    post = post[post['src']==CLIENT_IP]

    # Process raw timings
    pre = pd.read_csv(fname_keys, index_col=0).sort_values('press_time')
    pre['press_time'] = pre['press_time'].values/1e6
    pre['release_time'] = pre['release_time'].values/1e6

    # Normalize time to the first emitted key press
    t0 = pre.iloc[0]['press_time']
    pre['press_time'] -= t0
    pre['release_time'] -= t0
    post['frame_time'] -= t0

    # Detect ground truth and keep either the http or http2 columns
    http1 = post[['http_path','http_method']]
    http2 = post[['http2_path','http2_method']]
    http1.columns = ['path','method']
    http2.columns = ['path','method']
    del post['http_path']
    del post['http_method']
    del post['http2_path']
    del post['http2_method']

    if http1['path'].isnull().all() and http2['path'].isnull().all():
        print('Warning: no valid URL path to detect truth, skipping', name, flush=True)
        return False

    truth1, truth2 = [], []
    if (~http1['path'].isnull()).any():
        truth1 = detection_truth(pd.concat([post, http1], axis=1), website)

    if (~http2['path'].isnull()).any():
        truth2 = detection_truth(pd.concat([post, http2], axis=1), website)

    truth = truth1 if len(truth1) > len(truth2) else truth2

    if len(truth) == 0:
        print('Warning: no ground truth (ssl decryption probably failed), skipping', name, flush=True)
        return False

    if (truth['keys'].str.len() > MAX_KEYS_PER_QUERY).any():
        print('Warning: more than %d new keys in a single query, probably missing data' % MAX_KEYS_PER_QUERY, name, flush=True)
        return False

    if len(''.join(pre['key_name'].values)) != len(''.join(truth['keys'].values)):
        print('Warning: ground truth length does not match keystrokes, skipping', name, flush=True)
        return False

    if ''.join(pre['key_name'].values) != ''.join(truth['keys'].values):
        print('Warning: keystroke sequences differ, skipping', name, flush=True)
        return False

    # if (truth['keys'].str.len() > 1).sum()/len(truth) > MAX_RATIO_MULTI_KEYS:
    #     print('Warning: more than %f queries have multiple keys, probably missing data' % MAX_RATIO_MULTI_KEYS, name, flush=True)
    #     return False

    pre.to_csv(fname_pre)
    post.to_csv(fname_post)
    truth.to_csv(fname_truth)

    return True


if __name__ == '__main__':
    if len(sys.argv) > 1:
        fnames = sys.argv[1:]
    else:
        fnames = glob(os.path.join(DATA_DIR, 'capture','pcap', '*.pcap'))

    for fname_pcap in fnames:
        try:
            name = os.path.splitext(os.path.split(fname_pcap)[1])[0]

            if SKIP_FINISHED and capture_exists(name):
                print('Skipping', name, flush=True)
                continue

            print('Converting', fname_pcap, flush=True)
            success = convert_pcap(name)

            if MOVE_FAILURES and not success:
                failed_capture(name)

        except KeyboardInterrupt:
            print('Interrupted, quitting')
            break

    print('Done')
