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

from dotenv import dotenv_values

from selenium.webdriver.common.by import By

import mysql.connector as connector
from mysql.connector import errorcode


def send_report(send_to: dict):
    envs = dotenv_values('/home/server_bot/.env')
    avtobot = telegram.Bot(token=envs['AVTOBOT_TELEGRAM_TOKEN'])
    avtobot_chat_id = envs['AVTOBOT_CHAT_ID']

    products = list(send_to.keys())
    msg_text = ''

    # prepare for CSV file
    list_keys = ['unix_timestamp',
                 'avto_ru_total', 'avto_ru_used', 'avto_ru_company',
                 'avito_avto_total', 'avito_avto_used', 'avito_avto_company',
                 'drom_total', 'drom_used', 'drom_company',
                 'sber_avto_total', 'sber_avto_used']
    cars_dict = dict.fromkeys(list_keys, 0)

    cars_dict['unix_timestamp'] = str(int(time.time()))

    # iterate over all products (avto_ru, avito_avto, drom, sber_avto)
    for product in products:
        total_cars = send_to[product]['total']
        used_cars = send_to[product]['used']

        # SBER has no company and private categories
        company_cars = send_to[product].get('company', 'no_cars')

        # verify results; we can get no response from a website
        new_cars = 'no_cars' if total_cars == 'no_cars' or used_cars == 'no_cars' \
            else total_cars - used_cars

        private_cars = 'no_cars' if total_cars == 'no_cars' or company_cars == 'no_cars' \
            else total_cars - company_cars

        product_info = f'{product.upper()}\n total: {total_cars}\n used: {used_cars}\n new: {new_cars}\n' \
                       f' company: {company_cars}\n private: {private_cars}\n\n'

        msg_text += product_info

        # dict for CSV file
        cars_dict[f'{product}_total'] = total_cars
        cars_dict[f'{product}_used'] = used_cars
        cars_dict[f'{product}_company'] = company_cars

    # add index, only one row
    save_filename = '/home/server_bot/git_workspace/avto_server/avto_analytics.csv'
    df_results = pd.DataFrame(cars_dict,
                              index=[0])

    if not os.path.exists(f'{save_filename}'):
        df_results.to_csv(f'{save_filename}',
                          index=False,
                          mode='w')
    else:
        df_results.to_csv(f'{save_filename}',
                          index=False,
                          header=None,
                          mode='a')

    # insert into MySQL
    db_config = {
        'user': envs['MYSQL_USER'],
        'password': envs['MYSQL_PASSWORD'],
        'host': envs['MYSQL_HOST'],
        'database': envs['MYSQL_DB'],
        'port': envs['MYSQL_PORT']
    }

    try:
        cnx = connector.connect(**db_config)

        with cnx.cursor() as cursor:
            query_str = ("INSERT INTO avto_daily_stats "
                         "(unix_timestamp, avto_ru_total, avto_ru_used, avto_ru_company, "
                         "avito_avto_total, avito_avto_used, avito_avto_company, "
                         "drom_total, drom_used, drom_company, "
                         "sber_avto_total, sber_avto_used) "
                         "VALUES(%(unix_timestamp)s, %(avto_ru_total)s, %(avto_ru_used)s, %(avto_ru_company)s, "
                         "%(avito_avto_total)s, %(avito_avto_used)s, %(avito_avto_company)s, "
                         "%(drom_total)s, %(drom_used)s, %(drom_company)s, "
                         "%(sber_avto_total)s, %(sber_avto_used)s)")

            # cars_dict - dict from CSV
            cursor.execute(query_str, cars_dict)
            cnx.commit()

        cnx.close()

        # send to Telegram
        avtobot.send_message(chat_id=avtobot_chat_id, text=msg_text)

    except connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print('access denied')
        else:
            print(err)

        avtobot.send_message(chat_id=avtobot_chat_id, text=f'MYSQL error\n\n{err}')


def get_avto_ru_results(urls: dict) -> dict:
    # add options to run in bg
    options = webdriver.ChromeOptions()
    options.add_argument('--enable-javascript')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--start-maximized')

    # parsing from mobile
    options.add_argument("user-agent=userAgent=Mozilla/5.0"
                         + "(iPhone; CPU iPhone OS 15_4 like Mac OS X) AppleWebKit/605.1.15 "
                         + "(KHTML, like Gecko) CriOS/101.0.4951.44 Mobile/15E148 Safari/604.1")
    options.add_argument('--headless=new')

    # start browser with options
    browser = webdriver.Chrome(options=options)

    # captcha click, not images, just "I am not a robot"
    # try:
    #     captcha_checkbox = browser.find_element(By.ID, 'js-button')
    #     captcha_checkbox.click()
    # except Exception as e:
    #     pass
    #
    # time.sleep(5)

    results_dict = {}
    dict_keys = list(urls.keys())

    for key in dict_keys:
        browser.get(urls[key])
        html_text = browser.page_source
        browser.add_cookie({"name": "gradius", "value": "0", 'sameSite': 'Lax'})
        browser.get(urls[key])
        resp_list = re.findall(r'SortTabs__count\D*\d{1,3}\D*\d{1,3}.{1,10}предло', html_text)
        if len(resp_list) != 0:
            result = int(''.join(re.findall(r'\d+', resp_list[0])))
        else:
            result = 'no_cars'

        results_dict[key] = result

    browser.quit()

    return results_dict


def get_avito_avto_results(urls: dict) -> dict:
    # add options to run in bg
    options = webdriver.ChromeOptions()
    options.add_argument('--enable-javascript')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--start-maximized')

    # parsing from mobile
    # options.add_argument("user-agent=userAgent=Mozilla/5.0"
    #                      + "(iPhone; CPU iPhone OS 15_4 like Mac OS X) AppleWebKit/605.1.15 "
    #                      + "(KHTML, like Gecko) CriOS/101.0.4951.44 Mobile/15E148 Safari/604.1")

    options.add_argument('--headless=new')

    # start browser with options
    browser = webdriver.Chrome(options=options)

    results_dict = {}
    dict_keys = list(urls.keys())

    for key in dict_keys:
        browser.get(urls[key])
        html_text = browser.page_source
        try:
            resp_list = re.findall(r'>\d{1,3}\S*\d{1,3}<',
                                   re.findall(r'<span.*count\S*>\d{1,3}\S*\d{1,3}</span>',
                                              html_text)[0])
            result = int(''.join(re.findall(r'\d+', resp_list[0])))
        except Exception as e:
            result = 'no_cars'

        results_dict[key] = result

    browser.quit()

    return results_dict


def get_drom_results(urls: dict) -> dict:
    # add options to run in bg
    options = webdriver.ChromeOptions()
    options.add_argument('--enable-javascript')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--start-maximized')

    # parsing from mobile
    options.add_argument("user-agent=userAgent=Mozilla/5.0"
                         + "(iPhone; CPU iPhone OS 15_4 like Mac OS X) AppleWebKit/605.1.15 "
                         + "(KHTML, like Gecko) CriOS/101.0.4951.44 Mobile/15E148 Safari/604.1")

    options.add_argument('--headless=new')

    # start browser with options
    browser = webdriver.Chrome(options=options)

    results_dict = {}
    dict_keys = list(urls.keys())

    for key in dict_keys:
        browser.get(urls[key])
        html_text = browser.page_source

        resp_list = re.findall(r'>\d{0,2}\D{,10}\d{1,3}\D{,2}объяв', html_text)

        if len(resp_list) != 0:
            result = int(''.join(re.findall(r'\d+', resp_list[0])))
        else:
            result = 'no_cars'

        results_dict[key] = result

    browser.quit()

    return results_dict


def get_sber_avto_results(urls: dict) -> dict:
    # add options to run in bg
    options = webdriver.ChromeOptions()
    options.add_argument('--enable-javascript')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--start-maximized')

    # parsing from mobile
    # options.add_argument("user-agent=userAgent=Mozilla/5.0"
    #                      + "(iPhone; CPU iPhone OS 15_4 like Mac OS X) AppleWebKit/605.1.15 "
    #                      + "(KHTML, like Gecko) CriOS/101.0.4951.44 Mobile/15E148 Safari/604.1")

    options.add_argument('--headless=new')

    # start browser with options
    browser = webdriver.Chrome(options=options)

    results_dict = {}
    dict_keys = list(urls.keys())

    for key in dict_keys:
        browser.get(urls[key])

        # faster than WebWait
        time.sleep(3)

        html_text = browser.page_source

        response_no_space = re.sub(r'\s+', '', string=html_text)
        resp_list = re.findall(r'Amount\D*\d{1,3}\D*\d{1,3}.{,10}предлож', response_no_space)

        if len(resp_list) != 0:
            result = int(''.join(re.findall(r'\d+', resp_list[0])))
        else:
            result = 'no_cars'

        results_dict[key] = result

    browser.quit()

    return results_dict


# spb, geo 0 km
avito_avto = 'https://www.avito.ru/sankt-peterburg/avtomobili?cd=1&radius=0&s=104&searchRadius=0'
avto_ru = 'https://auto.ru/sankt-peterburg/cars/all/'
drom = 'https://spb.drom.ru/auto/all/'
sber_avto = 'https://sberauto.com/sankt-peterburg/cars?g' \
            'eoDistance=0&rental_car=exclude_rental&isCreditSearchEnabled=false' \
            '&isWarrantySearchEnabled=false&isNoMileageInRussiaEnabled=false'

# used cars
avito_avto_used = 'https://www.avito.ru/sankt-peterburg/avtomobili/s_probegom-ASgBAgICAUSGFMjmAQ?'\
                  'f=ASgBAgICAkSGFMjmAfrwD~i79wI&radius=0&s=104&searchRadius=0'
avto_ru_used = 'https://auto.ru/sankt-peterburg/cars/used/'
drom_used = 'https://spb.drom.ru/auto/used/all/'
sber_avto_used = 'https://sberauto.com/sankt-peterburg/cars/used?geoDistance=0&isNew=false'\
                 '&rental_car=exclude_rental&isCreditSearchEnabled=false&isWarrantySearchEnabled=false'\
                 '&isNoMileageInRussiaEnabled=false'

# private and company cars
avito_avto_company = 'https://www.avito.ru/sankt-peterburg/avtomobili?radius=0&s=104&searchRadius=0&user=2'
avto_ru_company = 'https://auto.ru/sankt-peterburg/cars/all/?seller_group=COMMERCIAL'
drom_company = 'https://spb.drom.ru/auto/all/?owner_type=2'

# lists of urls
avto_ru_urls = {'total': avto_ru,
                'used': avto_ru_used,
                'company': avto_ru_company}
avito_avto_urls = {'total': avito_avto,
                   'used': avito_avto_used,
                   'company': avito_avto_company}
drom_urls = {'total': drom,
             'used': drom_used,
             'company': drom_company}
sber_avto_urls = {'total': sber_avto,
                  'used': sber_avto_used}

# get total cars from each resource
avto_ru_results = get_avto_ru_results(urls=avto_ru_urls)
avito_avto_results = get_avito_avto_results(urls=avito_avto_urls)
drom_results = get_drom_results(urls=drom_urls)
sber_avto_results = get_sber_avto_results(urls=sber_avto_urls)

# send report to avtobot
dict_send_to = {'avto_ru': avto_ru_results,
                'avito_avto': avito_avto_results,
                'drom': drom_results,
                'sber_avto': sber_avto_results}
send_report(send_to=dict_send_to)
