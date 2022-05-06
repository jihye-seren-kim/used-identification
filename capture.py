import os
import sys
import time
import string
import signal
import uinput
import subprocess
import numpy as np
import pandas as pd
from itertools import product
from selenium import webdriver
# from tbselenium import tbdriver
# from tbselenium import common as tbcm

from pcap2csv import convert_pcap
from common import *

BROWSERS = [
    'chrome',
    'firefox'
]

WEBSITES = {
    'google': 'https://www.google.com',
    # 'bing': 'https://www.bing.com',
    # 'duckduckgo': 'https://start.duckduckgo.com',
    'baidu': 'https://www.baidu.com',
    # 'yandex': 'https://www.yandex.com',
    # 'amazon': 'https://www.amazon.com',
    # 'wikipedia': 'https://www.wikipedia.com',
    # '360search': 'https://www.so.com',
}

KB_DEVICE = uinput.Device(uinput._CHAR_MAP.values())

KEY_PRESS = 1
KEY_RELEASE = 0

TSHARK_WEB = 'tshark -w %s -f "(tcp dst port 443 or tcp dst port 80 or tcp src port 443 or tcp src port 80)"'


def emit_events(device, events, sleep_interval=0.0001):
    assert len(events) > 0

    t0, e, v = events[0]
    device.emit(e, v, syn=True)
    t0_actual = time.time_ns()
    actual = [t0_actual]

    for t, e, v in events[1:]:
        next_t = t0_actual + (t - t0)
        while time.time_ns() < next_t:
            time.sleep(sleep_interval)

        device.emit(e, v, syn=True)
        # Record the actual time the event was emitted
        actual.append(time.time_ns())

    return actual


# def keystrokes2events(keystrokes):
#     press_actions = [(t, k, KEY_PRESS) for k, t, _ in keystrokes]
#     release_actions = [(t, k, KEY_RELEASE) for k, _, t in keystrokes]
#     events = sorted(press_actions + release_actions, key=itemgetter(0))
#     return events


def events2keystrokes(events):
    keysdown = {}
    keystrokes = []
    for t, k, action in events:
        if action == KEY_PRESS:
            keysdown[k] = t
        elif action == KEY_RELEASE and k in keysdown.keys():
            keystrokes.append((k, keysdown[k], t))
            del keysdown[k]
    return keystrokes


def make_driver(browser):
    # Start the web driver
    if browser == 'chrome':
        options = webdriver.ChromeOptions()
        options.add_argument("--disable-quic")
        driver = webdriver.Chrome(options=options)
    elif browser == 'firefox':
        driver = webdriver.Firefox()
    elif browser == 'opera':
        # TODO: bug, opera driver needs to be told where the binary is
        options = webdriver.ChromeOptions()
        options.add_argument("--disable-quic")
        options.binary_location = '/usr/bin/opera'
        opera_profile = '/home/vinnie/.config/opera'
        options.add_argument('user-data-dir=' + opera_profile)
        driver = webdriver.Opera(options=options)
    elif browser == 'torbrowser':
        tbpath = '/home/vinnie/.local/share/torbrowser/tbb/x86_64/tor-browser_en-US/'
        driver = tbdriver.TorBrowserDriver(tbpath, tor_cfg=tbcm.USE_RUNNING_TOR)
    else:
        raise Exception('Unknown browser:', browser)
    return driver


def capture_keystrokes_web(name, events, events_per_page=None,
                           start_delay=2, capture_delay=1, finish_delay=1):
    if events_per_page is None:
        events_per_page = len(events)

    browser, website, trial = name.split('_')
    fname_keys, fname_pcap, fname_ssl = capture_raw_fnames(name)

    # Create the driver
    os.environ['SSLKEYLOGFILE'] = fname_ssl
    driver = make_driver(browser)

    cap = None
    try:
        # Clear cookies
        driver.delete_all_cookies()

        # Start network capture
        cap = subprocess.Popen(TSHARK_WEB % fname_pcap,  shell=True, preexec_fn=os.setsid,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        time.sleep(capture_delay)

        for i in range(0, len(events), events_per_page):
            events_page = events[i:(i+events_per_page)]

            # Get the web page
            driver.get(WEBSITES[website])

            # Need to request focus for amazon
            if website == 'amazon':
                driver.find_element_by_id('twotabsearchtextbox').click()

            # Allow time for the website to load and uinput device to create
            time.sleep(start_delay)

            # Emit the keystrokes
            pre_times = emit_events(KB_DEVICE, events_page)

            # Let the browser finish transmitting
            time.sleep(finish_delay)

        success = True
    except Exception:
        success = False
    finally:
        # End capture
        if cap is not None:
            os.killpg(os.getpgid(cap.pid), signal.SIGTERM)

        driver.quit()

    if success:
        # Save the ground truth keystrokes emitted
        _, keys, actions = zip(*events)
        pre_events = list(zip(pre_times, keys, actions))
        pre_keystrokes = events2keystrokes(pre_events)
        df = pd.DataFrame(pre_keystrokes, columns=['key_name','press_time','release_time'])
        df = df.sort_values('press_time').reset_index(drop=True)
        df.index.name = 'idx'
        df.to_csv(fname_keys)

    return success


def capture_product(capture_type, trials):
    device = uinput.Device(uinput._CHAR_MAP.values())

    for trial, keystrokes in trials:
        for browser, website in product(BROWSERS, WEBSITES.keys()):
            if capture_raw_exists(name):
                continue

            name = '%s_%s_%s_%s' % (capture_type, browser, website, trial)

            print('Capture:', name)
            res = capture_keystrokes_web(name, keystrokes)

            if not res:
                print('Failed:', name)

def df2keystrokes(df, segment):
    return df.xs(segment, level='segment')[['key_name','press_time','release_time']].to_records(index=False).tolist()


def capture_until_success(trials):
    work = [capture_name(browser, website, trial) for
            browser, website, trial in product(BROWSERS, WEBSITES.keys(), trials.keys())]

    num_capture_attempts = 0
    num_capture_failures = 0
    num_convert_failures = 0
    while len(work) > 0:
        name = work.pop(0)

        if capture_raw_exists(name):
            continue

        print('Remaining captures:', len(work), flush=True)
        num_capture_attempts += 1

        print('Capture:', name, flush=True)
        trial = capture_attributes(name)['trial']
        res = capture_keystrokes_web(name, trials[trial])

        if not res:
            print('Failed capture:', name, flush=True)
            work.append(name)
            failed_capture(name)
            num_capture_failures += 1
            continue

        print('Convert:', name, flush=True)
        res = convert_pcap(name)

        if not res:
            print('Failed convert:', name, flush=True)
            work.append(name)
            failed_capture(name)
            num_convert_failures += 1
            continue

    print('Done', flush=True)
    print('Capture attempts:', num_capture_attempts, flush=True)
    print('Capture failures:', num_capture_failures, flush=True)
    print('Convert failures:', num_convert_failures, flush=True)


def keystrokes2events(df, char='a'):
    press = df[['press_time','key_code']]
    release = df[['release_time','key_code']]
    press.columns = ['time','key_code']
    release.columns = ['time','key_code']
    press['action'] = KEY_PRESS
    release['action'] = KEY_RELEASE

    events = pd.concat([press,release])
    events = events.sort_values('time')
    events['time'] = events['time']*1000000

    if char is not None:
        events['key_code'] = [uinput._CHAR_MAP.get(char)]*len(events)

    events = events[['time','key_code','action']].to_records(index=False).tolist()
    return events


def main(dataset):
    df = load_data(dataset)
    idx = df.index.to_frame()
    df.index = pd.Index(idx['user'].astype(str) + '-' + idx['session'].astype(str), name='user-session')
    trials = df.groupby('user-session').apply(keystrokes2events)
    trials = trials.to_dict()
    capture_until_success(trials)


if __name__ == '__main__':
    main('keystrokes/aalto-eval')
