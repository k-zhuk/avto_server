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
from dotenv import dotenv_values


def send_report(avto_ru: str, avito_avto: str,
                drom: str, sber_avto: str):

    envs = dotenv_values('/home/server_bot/.env')
    avtobot = telegram.Bot(token=envs['AVTOBOT_TELEGRAM_TOKEN'])

    msg_text = f'avto_ru: {avto_ru}\navito_avto: {avito_avto}'\
               f'\ndrom: {drom}\nsber_avto: {sber_avto}'

    avtobot.send_message(chat_id=158532925, text=msg_text)


def get_avto_ru_total_cars(webpage_url: str) -> str:
    # add options to run in bg
    options = webdriver.ChromeOptions()
    options.add_argument('--enable-javascript')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--start-maximized')
    options.add_argument("user-agent=userAgent=Mozilla/5.0"
                         + "(iPhone; CPU iPhone OS 15_4 like Mac OS X) AppleWebKit/605.1.15 "
                         + "(KHTML, like Gecko) CriOS/101.0.4951.44 Mobile/15E148 Safari/604.1")
    options.add_argument('--headless=new')

    # start browser with options
    browser = webdriver.Chrome(options=options)
    browser.get(webpage_url)

    # captcha click, not images, just "I am not a robot"
    # try:
    #     captcha_checkbox = browser.find_element(By.ID, 'js-button')
    #     captcha_checkbox.click()
    # except Exception as e:
    #     pass
    #
    # time.sleep(5)

    html_text = browser.page_source

    resp_list = re.findall(r'itemRadius.*Санкт\D*itemCount\D*\d{1,3}.{,10}\d{1,3}', html_text)

    if len(resp_list) != 0:
        avto_ru_total_cars = ''.join(re.findall(r'\d+', resp_list[0]))
    else:
        avto_ru_total_cars = 'no_cars'

    browser.quit()

    return avto_ru_total_cars


def get_avito_avto_total_cars(webpage_url: str) -> str:
    # add options to run in bg
    options = webdriver.ChromeOptions()
    options.add_argument('--enable-javascript')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--start-maximized')
    # options.add_argument("user-agent=userAgent=Mozilla/5.0"
    #                      + "(iPhone; CPU iPhone OS 15_4 like Mac OS X) AppleWebKit/605.1.15 "
    #                      + "(KHTML, like Gecko) CriOS/101.0.4951.44 Mobile/15E148 Safari/604.1")
    options.add_argument('--headless=new')

    # start browser with options
    browser = webdriver.Chrome(options=options)
    browser.get(webpage_url)

    html_text = browser.page_source

    try:
        resp_list = re.findall(r'>\d{1,3}\S*\d{1,3}<',
                               re.findall(r'<span.*count\S*>\d{1,3}\S*\d{1,3}</span>',
                                          html_text)[0])

        avito_avto_total_cars = ''.join(re.findall(r'\d+', resp_list[0]))
    except Exception as e:
        avito_avto_total_cars = 'no_cars'

    browser.quit()

    return avito_avto_total_cars


def get_drom_total_cars(webpage_url: str) -> str:
    # add options to run in bg
    options = webdriver.ChromeOptions()
    options.add_argument('--enable-javascript')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--start-maximized')
    options.add_argument("user-agent=userAgent=Mozilla/5.0"
                         + "(iPhone; CPU iPhone OS 15_4 like Mac OS X) AppleWebKit/605.1.15 "
                         + "(KHTML, like Gecko) CriOS/101.0.4951.44 Mobile/15E148 Safari/604.1")
    options.add_argument('--headless=new')

    # start browser with options
    browser = webdriver.Chrome(options=options)

    browser.get(webpage_url)
    html_text = browser.page_source

    resp_list = re.findall(r'\d{0,2}\D{,10}\d{1,3}\D{,2}объяв', html_text)

    if len(resp_list) != 0:
        drom_total_cars = ''.join(re.findall(r'\d+', resp_list[0]))
    else:
        drom_total_cars = 'no_cars'

    browser.quit()

    return drom_total_cars


def get_sber_avto_total_cars(webpage_url: str) -> str:
    # add options to run in bg
    options = webdriver.ChromeOptions()
    options.add_argument('--enable-javascript')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--start-maximized')
    options.add_argument("user-agent=userAgent=Mozilla/5.0"
                         + "(iPhone; CPU iPhone OS 15_4 like Mac OS X) AppleWebKit/605.1.15 "
                         + "(KHTML, like Gecko) CriOS/101.0.4951.44 Mobile/15E148 Safari/604.1")
    options.add_argument('--headless=new')

    # start browser with options
    browser = webdriver.Chrome(options=options)

    browser.get(webpage_url)
    html_text = browser.page_source

    response_no_space = re.sub(r'\s+', '', string=html_text)
    resp_list = re.findall(r'Amount\D*\d{1,3}\D*\d{1,3}.{,10}предлож', response_no_space)

    if len(resp_list) != 0:
        sber_avto_total_cars = ''.join(re.findall(r'\d+', resp_list[0]))
    else:
        sber_avto_total_cars = 'no_cars'

    browser.quit()

    return sber_avto_total_cars


# spb, geo 0 km
avito_avto = 'https://www.avito.ru/sankt-peterburg/avtomobili?cd=1&radius=0&s=104&searchRadius=0'
avto_ru = 'https://auto.ru/sankt-peterburg/cars/all/'
drom = 'https://spb.drom.ru/auto/all/'
sber_avto = 'https://sberauto.com/sankt-peterburg/cars?g' \
            'eoDistance=0&rental_car=exclude_rental&isCreditSearchEnabled=false' \
            '&isWarrantySearchEnabled=false&isNoMileageInRussiaEnabled=false'

# get total cars from each resource
avto_ru_total_cars = get_avto_ru_total_cars(webpage_url=avto_ru)
avito_avto_total_cars = get_avito_avto_total_cars(webpage_url=avito_avto)
drom_total_cars = get_drom_total_cars(webpage_url=drom)
sber_avto_total_cars = get_sber_avto_total_cars(webpage_url=sber_avto)

# send report to avtobot
send_report(avto_ru=avto_ru_total_cars, avito_avto=avito_avto_total_cars,
            drom=drom_total_cars, sber_avto=sber_avto_total_cars)

