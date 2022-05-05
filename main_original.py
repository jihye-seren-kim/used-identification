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

URL = ['www.google.com', 'www.youtube.com', 'www.instagram.com', 'www.bing.com', 'www.facebook.com', 'www.duckduckgo.com', 'www.amazon.com', ]

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath('__file__')))

_display = Display(os.environ['DISPLAY'])

ALL_INPUTS = [
'button',
'checkbox',
'color',
'date',
'datetime-local',
'email',
'file',
'hidden',
'image',
'month',
'number',
'password',
'reset',
'search',
'submit',
'tel',
'text',
'time',
'url',
'week',
]

TEXT_INPUTS = [
'number',
'text',
'search',
]

CLICK_INPUTS = [
'submit',
'button',
'checkbox',
'radio',
'input',
]


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

           

def browse(driver, bar_height=115):

    def centered():
        return {'x': page2screen(e.rect)['x'] + e.rect['width']/2, 'y': page2screen(e.rect)['y']}
    
    def page_rect(driver=driver):
        return {
            'x':driver.execute_script('return window.pageXOffset;'), 
            'y':driver.execute_script('return window.pageYOffset;'),
            'height':driver.execute_script('return window.innerHeight;'),
            'width':driver.execute_script('return window.innerWidth;'),
        }

    def window_rect(driver=driver):
        return driver.get_window_rect()

    def page2screen(rect):
        try:
            r = rect.copy()
            w = window_rect()
            p = page_rect()
            r['x'] = r['x'] + w['x'] - p['x']
            r['y'] = r['y'] + w['y'] + bar_height - p['y']
            return r
        except TypeError:
            pass

    def location_check():
        x = mouse_loc()['x']
        y = mouse_loc()['y']
        if (x >= page_rect(driver=driver)['width']) or (y >= page_rect(driver=driver)['height']): 
            x = page_rect(driver=driver)['width']/2
            y = page_rect(driver=driver)['height']/2
            return x, y
        else: 
            return x, y

                  
    # 1. move the mouse over random images (<10) 
    for new_page in range(3):      
        try:
            work = [e for e in driver.find_elements_by_xpath(f"//img") if e.is_displayed()]
            random.shuffle(work)
            work = work[0:10]
            visited = set()
            while len(work) > 0:
                e = work.pop(0)
                visited.add(e._id)                    
                print("mouse movement(images), ", "time:", time.time()*1000, ",", mouse_loc(), ",", e._id, ",", e.get_attribute('name'), ",", e.get_attribute('id'), ",", e.get_attribute('alt'), ",", e.get_attribute('src'), file=log_move)
                mouse_scroll_to(e.rect, page_rect)
                mouse_move_to(page2screen(e.rect)['x']+e.rect['width']/2, page2screen(e.rect)['y'])
                try: 
                    coord_x, coord_y = location_check()
                    mouse_move_to(coord_x, coord_y)
                except TypeError:
                    pass
                time.sleep(random.uniform(0.1, 0.2))                                                
        except exceptions.StaleElementReferenceException as e:
            pass
                            
    # 2. move the mouse to five random inputs
        try:
            work = [e for e in driver.find_elements_by_xpath(f"//input") if e.is_displayed()]
            random.shuffle(work)
            work = work[0:5]        
            visited = set()
            while len(work) > 0:
                e = work.pop(0)
                visited.add(e._id)
                print("mouse movement(inputs), ", "time:", time.time()*1000, ",", mouse_loc(), ",", e._id, ",", e.get_attribute('name'), ",", e.get_attribute('id'), ",",  e.get_attribute('alt'), file=log_move)
                mouse_scroll_to(e.rect, page_rect)
                mouse_move_to(page2screen(e.rect)['x']+e.rect['width']/2, page2screen(e.rect)['y'])                          
                try: 
                    coord_x, coord_y = location_check()
                    mouse_move_to(coord_x, coord_y)
                except TypeError:
                    pass
                time.sleep(random.uniform(0.1, 0.2))                       
        except exceptions.StaleElementReferenceException as e:
                pass

    # 3. type in text inputs
        if (new_page == 0): 
            for input_type in TEXT_INPUTS:
                try:
                    work = [e for e in driver.find_elements_by_xpath(f"//input[@type='{input_type}']") if e.is_displayed()]
                    random.shuffle(work)
                    work[0:2]
                    visited = set()
                    while len(work) > 0:
                        e = work.pop(0)
                        visited.add(e._id)
                        mouse_move_to(page2screen(e.rect)['x']+e.rect['width']/2, page2screen(e.rect)['y'])
                        try: 
                            coord_x, coord_y = location_check()
                            mouse_move_to(coord_x, coord_y)
                        except TypeError:
                            pass
                        ag.click()
                        print("mouse click, ", "time:", time.time()*1000, ",", mouse_loc(), ",", e._id, ",", e.get_attribute('name'), ",", e.get_attribute('id'), file=log_click) 
                        key_type()
                except exceptions.StaleElementReferenceException as e:
                    pass    
 
            for input_type in CLICK_INPUTS:            
                try:            
                    work = [e for e in driver.find_elements_by_xpath(f"//input[@type='{input_type}']") if e.is_displayed()]
                    work[0:2]                    
                    visited = set()     
                    while len(work) > 0:
                        e = work.pop(0)
                        visited.add(e._id)
                        mouse_move_to(page2screen(e.rect)['x']+e.rect['width']/2, page2screen(e.rect)['y'])
                        try: 
                            coord_x, coord_y = location_check()
                            mouse_move_to(coord_x, coord_y)
                        except TypeError:
                            pass                          
                        ag.click()                       
                        print("mouse click, ", "time:", time.time()*1000, ",", mouse_loc(), ",", e._id, ",", e.get_attribute('name'), ",", e.get_attribute('id'), file=log_click)                     
                except exceptions.StaleElementReferenceException as e:
                    pass

                    
        elif (new_page > 0):
            try:
                for input_type in ALL_INPUTS:
                    work = [e for e in driver.find_elements_by_xpath(f"//input[@type='{input_type}']") if e.is_displayed()] 
                    #from IPython import embed; embed()                
                    random.shuffle(work)
                    work = work[0:5]        
                    visited = set()
                    while len(work) > 0:
                        e = work.pop(0)
                        visited.add(e._id)
                        print("mouse movement(inputs), ", "time:", time.time()*1000, ",", mouse_loc(), ",", e._id, ",", e.get_attribute('name'), ",", e.get_attribute('id'), ",",  e.get_attribute('alt'), file=log_move)
                        mouse_scroll_to(e.rect, page_rect)
                        mouse_move_to(page2screen(e.rect)['x']+e.rect['width']/2, page2screen(e.rect)['y'])
                        try: 
                            coord_x, coord_y = location_check()
                            mouse_move_to(coord_x, coord_y)
                        except TypeError:
                            pass                         
                        time.sleep(random.uniform(0.1, 0.2))                       
            except exceptions.StaleElementReferenceException as e:
                pass

            try:
                work_1 = [e for e in driver.find_elements_by_xpath(f"//input") if e.is_displayed()]
                work_2 = [e for e in driver.find_elements_by_xpath(f"//img") if e.is_displayed()]
                #from IPython import embed; embed()                
                work = work_1 + work_2
                random.shuffle(work)
                work = work[0:2]        
                visited = set()
                while len(work) > 0:
                    e = work.pop(0)
                    visited.add(e._id)
                    print("mouse movement(inputs), ", "time:", time.time()*1000, ",", mouse_loc(), ",", e._id, ",", e.get_attribute('name'), ",", e.get_attribute('id'), ",",  e.get_attribute('alt'), file=log_move)
                    mouse_scroll_to(e.rect, page_rect)
                    mouse_move_to(page2screen(e.rect)['x']+e.rect['width']/2, page2screen(e.rect)['y'])
                    try: 
                        coord_x, coord_y = location_check()
                        mouse_move_to(coord_x, coord_y)
                    except TypeError:
                        pass                         
                    time.sleep(random.uniform(0.1, 0.2))
                    ag.click()                       
            except exceptions.StaleElementReferenceException as e:
                pass
                               
               
                                                  
def main(url):       
    # open the browser  
    cap = None

    #profile = webdriver.FirefoxProfile("/home/ubuntu/.mozilla/firefox/qpdx5l2k.default-release")
    profile = webdriver.FirefoxProfile("/home/jihyekim/.mozilla/firefox/f6u4hokh.default-release")
    driver = webdriver.Firefox(firefox_profile=profile)
    driver.maximize_window()
    #driver.set_window_size(1200,900)

    
   #TODO: start traffic capture        
    cap = subprocess.Popen(TSHARK_WEB % pcap, shell=True, preexec_fn=os.setsid, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(random.uniform(0.5, 1))
       
    # load a URL
    driver.get(url)
    time.sleep(random.uniform(1, 2))

    # start interactions
    browse(driver)        
    driver.quit()
                        
    if cap is not None:
        os.killpg(os.getpgid(cap.pid), signal.SIGTERM)


if __name__ == '__main__':
    os.environ['DISPLAY'] = ':0'
    for i in range(len(URL)):
        url = "https://" + URL[i] 
        adr = url.split(".")[1]
        pcap = os.path.join(ROOT_DIR, 'capture', 'pcaps', adr + '.pcap')
        log_click = open(os.path.join(ROOT_DIR, 'capture', 'inputs', 'click', adr + '.csv'), "w") 
        log_type = open(os.path.join(ROOT_DIR, 'capture', 'inputs', 'typing', adr + '.csv'), "w") 
        log_move = open(os.path.join(ROOT_DIR, 'capture', 'inputs', 'movement', adr + '.csv'), "w")
        log_scroll = open(os.path.join(ROOT_DIR, 'capture', 'inputs', 'scroll', adr + '.csv'), "w")
        main(url)
        log_click.close()
        log_type.close()
        log_move.close()
        log_scroll.close()

