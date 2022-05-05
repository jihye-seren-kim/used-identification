from user_behavior import *

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


def browse(driver, bar_height=100):
    
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

               
    def work_repeat(idx_1, idx_2): 
           
        def work_choice(idx):
            work=[]    
            if (idx == 1):
                work = [e for e in driver.find_elements_by_xpath(f"//img") if e.is_displayed()]
                return work
            elif (idx == 2):
                work = [e for e in driver.find_elements_by_xpath(f"//input") if e.is_displayed()]
                return work  
            elif (idx == 3):
                work = [e for e in driver.find_elements_by_xpath(f"//input[@type='{input_type}']") if e.is_displayed()] 
                return work
              
                
        def print_out(idx):
            if (idx == 1):
                print("images, ", "time:", time.time()*1000, ",", mouse_loc(), ",", e._id, ",", e.get_attribute('name'), ",", e.get_attribute('id'), ",", e.get_attribute('alt'), ",", e.get_attribute('src'), file=log_move)
            elif (idx == 2):
                print("inputs, ", "time:", time.time()*1000, ",", mouse_loc(), ",", e._id, ",", e.get_attribute('name'), ",", e.get_attribute('id'), ",",  e.get_attribute('alt'), file=log_move)
            elif (idx == 3):
                ag.click()
                print("mouse click, ", "time:", time.time()*1000, ",", mouse_loc(), ",", e._id, ",", e.get_attribute('name'), ",", e.get_attribute('id'), file=log_click) 
     
        try:
            work = work_choice(idx_1)
            #work = re.sub('\'', '', work) 
            random.shuffle(work)
            work = work[0:10]
            visited = set()
            while len(work) > 0:    
                e = work.pop(0)
                visited.add(e._id)
                mouse_scroll_to(e.rect, page_rect)
                mouse_move_to(page2screen(e.rect)['x']+e.rect['width']/2, page2screen(e.rect)['y']) 
                print_out(idx_2)
                coord_x, coord_y = location_check()
                mouse_move_to(coord_x, coord_y)
                time.sleep(random.uniform(0.1, 0.2))                                                
        except exceptions.StaleElementReferenceException as e:
            pass
        except TypeError:
            pass 


    for new_page in range(10):      
        work_repeat(1, 1)
        work_repeat(2, 2) 
        #from IPython import embed; embed()            
        if (new_page == 0):      
            for input_type in TEXT_INPUTS:
                work_repeat(3, 3)
                key_type()           
            for input_type in CLICK_INPUTS:            
                work_repeat(3, 3)
        elif (new_page > 0):
            for input_type in ALL_INPUTS:
                work_repeat(3, 2)                                  
            for input_type in CLICK_INPUTS:            
                work_repeat(3, 3)                      
                                               
def main(url):       
    cap = None 
    profile = webdriver.FirefoxProfile("/home/user/.mozilla/firefox/rav02ono.default-release")
    options = webdriver.FirefoxOptions()
    driver = webdriver.Firefox(firefox_profile=profile)
    #driver = webdriver.Firefox()
    driver.maximize_window()
    
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
    #os.environ['DISPLAY'] = ':0'
    pcap = os.path.join(ROOT_DIR, 'capture', 'pcaps', adr + '.pcap')
    main(url)
    log_click.close()
    log_type.close()
    log_move.close()
    log_scroll.close()
