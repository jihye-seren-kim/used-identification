import os
import time
import string
import signal
import random
import subprocess
import numpy as np
import pandas as pd
import pytweening 
from selenium import webdriver
from selenium.common import exceptions
import pyautogui as ag
from Xlib.display import Display
import Xlib.XK

TSHARK_WEB = 'tshark -w %s -f "(tcp dst port 443 or tcp dst port 80 or tcp src port 443 or tcp src port 80)"' 
_display = Display(os.environ['DISPLAY'])
ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath('__file__')))
#URL = ['www.google.com', 'www.youtube.com', 'www.amazon.com', 'www.cnn.com', 'www.instagram.com', 'www.bing.com', 'www.facebook.com', 'www.github.com']
url = 'https://www.instagram.com'
adr = url.split(".")[1]
        
log_click = open(os.path.join(ROOT_DIR, 'capture', 'inputs', 'click', adr + '.csv'), "w") 
log_type = open(os.path.join(ROOT_DIR, 'capture', 'inputs', 'typing', adr + '.csv'), "w") 
log_move = open(os.path.join(ROOT_DIR, 'capture', 'inputs', 'movement', adr + '.csv'), "w")
log_scroll = open(os.path.join(ROOT_DIR, 'capture', 'inputs', 'scroll', adr + '.csv'), "w")


def mouse_loc():
    (x,y) = ag.position()
    return {'x': x, 'y': y}
  

def mouse_move_to(x, y, duration=None, func=None):
    duration = np.random.uniform(2,3) 
    func = getattr(pytweening, 'linear')
    ag.moveTo(x, y, duration, func) 


def scroll_direction(w_rect, rect, x_pad, y_pad):
    delta = {'x': 0,'y': 0}
    try: 
        if rect['y'] < (w_rect['y'] + y_pad):
            delta['y'] = 1
        elif (rect['y'] + rect['height']) > (w_rect['y'] + w_rect['height'] - y_pad):
            delta['y'] = -1

        if rect['x'] < (w_rect['x'] + y_pad):
            delta['x'] = -1
        elif (rect['x'] + rect['width']) > (w_rect['x'] + w_rect['width'] - y_pad):
            delta['x'] = 1
    except:
        pass
    return delta


def mouse_scroll_to(rect, page_rect, x_pad=2, y_pad=2, sleep_interval=0.001, timeout=5):
    start_time = time.time()
    giveup = lambda: (time.time() - start_time) > timeout
    next_t = time.time_ns()
    prev_p_rect = page_rect()
    while not giveup():
        delta = scroll_direction(prev_p_rect, rect, x_pad, y_pad)
        if (delta['x'] == 0 and delta['y'] == 0):
            break
        while time.time_ns() < next_t:
            time.sleep(sleep_interval)
        next_t = next_t + 8000000
        ag.hscroll(delta['x'], x=None, y=None)
        print("horizontal scroll, ", "time:", time.time()*1000, ",", mouse_loc(), ",", file=log_scroll)   
      
        ag.vscroll(delta['y'], x=None, y=None)
        print( "vertical scroll, ", "time:", time.time()*1000, ",", mouse_loc(), ",", file=log_scroll)  
             
        p_rect = page_rect()
        prev_p_rect = p_rect
                         
                         
def key_type():
    keys = 'the quick brown fox 12345'
    for key in keys:
        time.sleep(random.uniform(0.3, 0.5))
        ag.typewrite(key)
        print("keyboard typing, ", "key: ", key, ", time:", time.time()*1000, file=log_type)

