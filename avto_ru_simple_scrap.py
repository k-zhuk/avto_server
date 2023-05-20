import numpy as np
import pandas as pd
import requests
import os
import shutil
import glob
import io
import re
from urllib.request import urlopen
import seaborn as sns
import matplotlib.pyplot as plt
import telegram
import time
from selenium import webdriver
from selenium.webdriver.common.by import By


def send_report(avto_ru: str):

    AVTOBOT_TELEGRAM_TOKEN = os.environ.get("AVTOBOT_TELEGRAM_TOKEN")
    avtobot = telegram.Bot(token=AVTOBOT_TELEGRAM_TOKEN)
    avtobot.send_message(chat_id=158532925, text=f'avto_ru: {avto_ru_total_cars}')

# spb, geo 0 km
avito_avto = 'https://www.avito.ru/sankt-peterburg/avtomobili?cd=1&radius=0&s=104&searchRadius=0'
avto_ru = 'https://auto.ru/sankt-peterburg/cars/all/'
drom = 'https://spb.drom.ru/auto/all/'
sber_avto = 'https://sberauto.com/sankt-peterburg/cars?g' \
            'eoDistance=0&rental_car=exclude_rental&isCreditSearchEnabled=false' \
            '&isWarrantySearchEnabled=false&isNoMileageInRussiaEnabled=false'

# add options to run in bg
options = webdriver.ChromeOptions()
options.add_argument('--headless=new')

# get html text
browser = webdriver.Chrome(options=options)
browser.get(avto_ru)
try:
    captcha_checkbox = browser.find_element(By.ID, 'js-button')
    captcha_checkbox.click()
except Exception as e:
    pass

time.sleep(5)
html_text = browser.page_source

resp_list = re.findall(r'itemRadius.*Санкт\D*itemCount\D*\d{1,3}.{,10}\d{1,3}', html_text)

if len(resp_list) != 0:
    avto_ru_total_cars = ''.join(re.findall(r'\d+', resp_list[0]))
else:
    avto_ru_total_cars = 'no cars'

send_report(avto_ru=avto_ru_total_cars)

browser.quit()
