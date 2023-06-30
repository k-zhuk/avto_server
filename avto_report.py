import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
import mysql.connector as connector
from mysql.connector import errorcode
from datetime import datetime
from datetime import timedelta
import scipy.stats as stats
import io
import telegram
from PyPDF2 import PdfWriter
from PyPDF2 import PdfReader
from dotenv import dotenv_values


def get_query_result(query: str) -> list:
    db_config = {
        'user': envs['MYSQL_USER'],
        'password': envs['MYSQL_PASSWORD'],
        'host': envs['MYSQL_HOST'],
        'database': envs['MYSQL_DB'],
        'port': envs['MYSQL_PORT']
    }

    try:
        cnx = connector.connect(**db_config)
    except connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print('access denied')
        else:
            print(err)

        return [[-1]]

    with cnx.cursor() as cursor:
        cursor.execute(query)
        query_results = cursor.fetchall()

    cnx.close()

    return query_results


def is_diff(arr_1: 'np.array', arr_2: 'np.array') -> tuple:
    # check if different of samples more than 10%
    if 0.4 <= len(arr_1) / (len(arr_1) + len(arr_2)) <= 0.6:

        arrs_dict = {}

        for i, arr in enumerate([arr_1, arr_2]):
            isnan_idx = np.isnan(arr)
            nan_idx = np.argwhere(isnan_idx).flatten()
            bootstrap_samples = arr[~isnan_idx]

            temp_list = []
            n_repeats = 10000

            for j in range(n_repeats):
                temp_list.append(np.mean(np.random.choice(bootstrap_samples, size=len(bootstrap_samples))))

            arrs_dict[f'arr_{i + 1}'] = [temp_list, nan_idx]

        statistic, p_value = stats.ttest_ind(a=arrs_dict['arr_1'][0],
                                             b=arrs_dict['arr_2'][0],
                                             equal_var=False)

        return p_value, arrs_dict['arr_1'][1], arrs_dict['arr_2'][1]
    else:
        return -1, -1, -1


def get_cohort(cohort: str, df_y_mean: pd.DataFrame, df_bfy_mean: pd.DataFrame) -> tuple:
    if cohort == 'company' or cohort == 'private':

        x_labels = [f'avto_ru_{cohort}',
                    f'avito_avto_{cohort}',
                    f'drom_{cohort}']
    else:
        x_labels = [f'avto_ru_{cohort}',
                    f'avito_avto_{cohort}',
                    f'drom_{cohort}',
                    f'sber_avto_{cohort}']

    total_cars = df_bfy_mean[x_labels].apply(int).to_frame(name='before_yesterday').reset_index() \
        .rename(columns={'index': 'product'})

    total_cars['yesterday'] = df_y_mean[x_labels].apply(int).values
    total_cars['cars_diff_percent'] = round(total_cars['yesterday'] / total_cars['before_yesterday'] - 1, 4) * 100

    product_labels = [f'+{round(x, 2)}%' if x >= 0 else f'{round(x, 2)}%' for x in total_cars['cars_diff_percent']]

    return product_labels, total_cars


def fill_no_cars(arr: 'array') -> tuple:
    new_cars_list = []
    private_cars_list = []

    for value in arr:
        # if 'no_cars' then set cars value to 0
        new_cars = value[0] - value[1] if value[0] != 0 and value[1] != 0 else 0
        new_cars_list.append(new_cars)

        private_cars = value[0] - value[2] if value[0] != 0 and value[2] != 0 else 0
        private_cars_list.append(private_cars)

    return new_cars_list, private_cars_list


def get_avto_ru(df_result: 'DataFrame') -> 'pdf bytes':
    # prepare data to PLOT
    before_yesterday_val = datetime.date(datetime.today()) - timedelta(days=2)
    yesterday_val = datetime.date(datetime.today()) - timedelta(days=1)
    today_val = datetime.date(datetime.today())

    df_before_yesterday = df_result.query('datetime_of_day.dt.date < @yesterday_val')
    df_yesterday = df_result.query('datetime_of_day.dt.date >= @yesterday_val')

    # zeros to NaN
    df_before_yesterday = df_before_yesterday.mask(df_before_yesterday == 0)
    df_yesterday = df_yesterday.mask(df_yesterday == 0)

    df_before_yesterday_init = df_before_yesterday.copy(deep=True)
    df_yesterday_init = df_yesterday.copy(deep=True)

    # get mean for legend
    df_before_yesterday_mean = df_before_yesterday.mean(numeric_only=True)
    df_yesterday_mean = df_yesterday.mean(numeric_only=True)

    # fillna
    df_before_yesterday = df_before_yesterday.fillna(df_before_yesterday.median(numeric_only=True))
    df_yesterday = df_yesterday.fillna(df_yesterday.median(numeric_only=True))

    sns.set_theme()
    fig, axes = plt.subplots(nrows=3,
                             ncols=2,
                             figsize=(23, 32),
                             facecolor='white')

    label_config = {'fontsize': '16'}
    title_config = {'fontsize': '24'}

    product_name = 'avto_ru'

    alpha_fill = 0.1
    alpha_edge = 0.87
    alpha_edge_shadow = 0.2

    x_ticks_labels = pd.date_range(start=before_yesterday_val, end=yesterday_val, freq='5T')[:-1]
    x_ticks_labels = list(map(lambda x: x.strftime('%H:%M'), x_ticks_labels))[::24] + list(
        x_ticks_labels[-1:].strftime('%H:%M'))

    # create ticks and add last element for plot
    x_ticks = np.arange(df_yesterday.shape[0])
    x_ticks = x_ticks[::24]
    x_ticks = np.insert(x_ticks, len(x_ticks), df_yesterday.shape[0])

    # =================================
    # Total cars
    # =================================

    legend_mean_val = int(df_yesterday_mean[f'{product_name}_total'])
    ax = sns.lineplot(ax=axes[0][0],
                      x=np.arange(df_yesterday.shape[0]),
                      y=df_yesterday[f'{product_name}_total'],
                      alpha=alpha_edge,
                      color='red',
                      label=f'yesterday ({legend_mean_val} cars)')

    # ===============================
    # draw the points with 'no_cars'
    p_value, idx_arr_bfy, idx_arr_y = is_diff(arr_1=df_before_yesterday_init[f'{product_name}_total'].values,
                                              arr_2=df_yesterday_init[f'{product_name}_total'].values)

    ax.scatter(idx_arr_bfy,
               df_before_yesterday[f'{product_name}_total'].values[idx_arr_bfy],
               marker='o',
               s=50,
               color='black',
               alpha=alpha_edge)

    ax.scatter(idx_arr_y,
               df_yesterday[f'{product_name}_total'].values[idx_arr_y],
               marker='o',
               color='red',
               alpha=alpha_edge,
               s=50)
    # ==============================

    legend_mean_val = int(df_before_yesterday_mean[f'{product_name}_total'])
    ax.plot(np.arange(df_before_yesterday.shape[0]),
            df_before_yesterday[f'{product_name}_total'],
            alpha=alpha_edge_shadow,
            color='black',
            label=f'yesterday - 1 ({legend_mean_val} cars)')

    ax.set_xlabel('Time', **label_config)
    ax.set_ylabel('Total cars', **label_config)
    ax.set_title(f'Total cars\np_value = {round(p_value, 4)}', **title_config)

    ax.set_xticks(ticks=x_ticks,
                  labels=x_ticks_labels,
                  rotation=45)
    ax.margins(x=0.01)

    y_lims = ax.get_ylim()

    ax.fill_between(x=df_yesterday['time_of_day'],
                    y1=y_lims[0],
                    y2=df_yesterday[f'{product_name}_total'],
                    alpha=alpha_fill,
                    color='red')

    # ax.get_figure().set_tight_layout('tight')
    ax.legend(fontsize=18)

    # =================================
    # Used cars
    # =================================

    legend_mean_val = int(df_yesterday_mean[f'{product_name}_used'])
    ax = sns.lineplot(ax=axes[1][0],
                      x=df_yesterday['time_of_day'],
                      y=df_yesterday[f'{product_name}_used'],
                      alpha=alpha_edge,
                      color='red',
                      ls='--',
                      label=f'yesterday ({legend_mean_val})')

    legend_mean_val = int(df_before_yesterday_mean[f'{product_name}_used'])
    ax.plot(np.arange(df_before_yesterday.shape[0]),
            df_before_yesterday[f'{product_name}_used'],
            alpha=alpha_edge_shadow,
            color='black',
            label=f'yesterday - 1 ({legend_mean_val} cars)')

    # ===============================
    # draw the points with 'no_cars'
    p_value, idx_arr_bfy, idx_arr_y = is_diff(arr_1=df_before_yesterday_init[f'{product_name}_used'].values,
                                              arr_2=df_yesterday_init[f'{product_name}_used'].values)

    ax.scatter(idx_arr_bfy,
               df_before_yesterday[f'{product_name}_used'].values[idx_arr_bfy],
               marker='o',
               s=50,
               color='black',
               alpha=alpha_edge)

    ax.scatter(idx_arr_y,
               df_yesterday[f'{product_name}_used'].values[idx_arr_y],
               marker='o',
               color='red',
               alpha=alpha_edge,
               s=50)
    # ==============================

    ax.set_xlabel('Time', **label_config)
    ax.set_ylabel('Used cars', **label_config)
    ax.set_title(f'Used cars\np_value = {round(p_value, 4)}', **title_config)

    # x_ticks = pd.concat([df_yesterday['time_of_day'].iloc[::24], df_yesterday['time_of_day'][-1:]])

    ax.set_xticks(ticks=x_ticks,
                  labels=x_ticks_labels,
                  rotation=45)
    ax.margins(x=0.01)

    ax.legend(fontsize=18)

    y_lims = ax.get_ylim()
    ax.fill_between(x=df_yesterday['time_of_day'],
                    y1=y_lims[0],
                    y2=df_yesterday[f'{product_name}_used'],
                    alpha=alpha_fill,
                    color='red')

    # =================================
    # New cars
    # =================================

    legend_mean_val = int(df_yesterday_mean[f'{product_name}_new'])
    ax = sns.lineplot(ax=axes[1][1],
                      x=df_yesterday['time_of_day'],
                      y=df_yesterday[f'{product_name}_new'],
                      alpha=alpha_edge,
                      color='red',
                      ls='--',
                      label=f'yesterday ({legend_mean_val})')

    legend_mean_val = int(df_before_yesterday_mean[f'{product_name}_new'])
    ax.plot(np.arange(df_before_yesterday.shape[0]),
            df_before_yesterday[f'{product_name}_new'],
            alpha=alpha_edge_shadow,
            color='black',
            label=f'yesterday - 1 ({legend_mean_val} cars)')

    # ===============================
    # draw the points with 'no_cars'
    p_value, idx_arr_bfy, idx_arr_y = is_diff(arr_1=df_before_yesterday_init[f'{product_name}_new'].values,
                                              arr_2=df_yesterday_init[f'{product_name}_new'].values)

    ax.scatter(idx_arr_bfy,
               df_before_yesterday[f'{product_name}_new'].values[idx_arr_bfy],
               marker='o',
               s=50,
               color='black',
               alpha=alpha_edge)

    ax.scatter(idx_arr_y,
               df_yesterday[f'{product_name}_new'].values[idx_arr_y],
               marker='o',
               color='red',
               alpha=alpha_edge,
               s=50)
    # ==============================

    ax.set_xlabel('Time', **label_config)
    ax.set_ylabel('New cars', **label_config)
    ax.set_title(f'New cars\np_value = {round(p_value, 4)}', **title_config)

    # x_ticks = pd.concat([df_yesterday['time_of_day'].iloc[::24], df_yesterday['time_of_day'][-1:]])

    ax.set_xticks(ticks=x_ticks,
                  labels=x_ticks_labels,
                  rotation=45)
    ax.margins(x=0.01)

    ax.legend(fontsize=18)

    y_lims = ax.get_ylim()
    ax.fill_between(x=df_yesterday['time_of_day'],
                    y1=y_lims[0],
                    y2=df_yesterday[f'{product_name}_new'],
                    alpha=alpha_fill,
                    color='red')

    # =================================
    # Company cars
    # =================================

    legend_mean_val = int(df_yesterday_mean[f'{product_name}_company'])
    ax = sns.lineplot(ax=axes[2][0],
                      x=df_yesterday['time_of_day'],
                      y=df_yesterday[f'{product_name}_company'],
                      alpha=alpha_edge,
                      color='red',
                      ls='-.',
                      label=f'yesterday ({legend_mean_val})')

    legend_mean_val = int(df_before_yesterday_mean[f'{product_name}_company'])
    ax.plot(np.arange(df_before_yesterday.shape[0]),
            df_before_yesterday[f'{product_name}_company'],
            alpha=alpha_edge_shadow,
            color='black',
            label=f'yesterday - 1 ({legend_mean_val} cars)')

    # ===============================
    # draw the points with 'no_cars'
    p_value, idx_arr_bfy, idx_arr_y = is_diff(arr_1=df_before_yesterday_init[f'{product_name}_company'].values,
                                              arr_2=df_yesterday_init[f'{product_name}_company'].values)

    ax.scatter(idx_arr_bfy,
               df_before_yesterday[f'{product_name}_company'].values[idx_arr_bfy],
               marker='o',
               s=50,
               color='black',
               alpha=alpha_edge)

    ax.scatter(idx_arr_y,
               df_yesterday[f'{product_name}_company'].values[idx_arr_y],
               marker='o',
               color='red',
               alpha=alpha_edge,
               s=50)
    # ==============================

    ax.set_xlabel('Time', **label_config)
    ax.set_ylabel('Company cars', **label_config)
    ax.set_title(f'Company cars\np_value = {round(p_value, 4)}', **title_config)

    # x_ticks = pd.concat([df_yesterday['time_of_day'].iloc[::24], df_yesterday['time_of_day'][-1:]])

    ax.set_xticks(ticks=x_ticks,
                  labels=x_ticks_labels,
                  rotation=45)
    ax.margins(x=0.01)

    ax.legend(fontsize=18)

    y_lims = ax.get_ylim()
    ax.fill_between(x=df_yesterday['time_of_day'],
                    y1=y_lims[0],
                    y2=df_yesterday[f'{product_name}_company'],
                    alpha=alpha_fill,
                    color='red')

    # =================================
    # Private cars
    # =================================

    legend_mean_val = int(df_yesterday_mean[f'{product_name}_private'])
    ax = sns.lineplot(ax=axes[2][1],
                      x=df_yesterday['time_of_day'],
                      y=df_yesterday[f'{product_name}_private'],
                      alpha=alpha_edge,
                      color='red',
                      ls='-.',
                      label=f'yesterday ({legend_mean_val})')

    legend_mean_val = int(df_before_yesterday_mean[f'{product_name}_private'])
    ax.plot(np.arange(df_before_yesterday.shape[0]),
            df_before_yesterday[f'{product_name}_private'],
            alpha=alpha_edge_shadow,
            color='black',
            label=f'yesterday - 1 ({legend_mean_val} cars)')

    # ===============================
    # draw the points with 'no_cars'
    p_value, idx_arr_bfy, idx_arr_y = is_diff(arr_1=df_before_yesterday_init[f'{product_name}_private'].values,
                                              arr_2=df_yesterday_init[f'{product_name}_private'].values)

    ax.scatter(idx_arr_bfy,
               df_before_yesterday[f'{product_name}_private'].values[idx_arr_bfy],
               marker='o',
               s=50,
               color='black',
               alpha=alpha_edge)

    ax.scatter(idx_arr_y,
               df_yesterday[f'{product_name}_private'].values[idx_arr_y],
               marker='o',
               color='red',
               alpha=alpha_edge,
               s=50)
    # ==============================

    ax.set_xlabel('Time', **label_config)
    ax.set_ylabel('Private cars', **label_config)
    ax.set_title(f'Private cars\np_value = {round(p_value, 4)}', **title_config)

    # x_ticks = pd.concat([df_yesterday['time_of_day'].iloc[::24], df_yesterday['time_of_day'][-1:]])

    ax.set_xticks(ticks=x_ticks,
                  labels=x_ticks_labels,
                  rotation=45)
    ax.margins(x=0.01)

    ax.legend(fontsize=18)

    y_lims = ax.get_ylim()
    ax.fill_between(x=df_yesterday['time_of_day'],
                    y1=y_lims[0],
                    y2=df_yesterday[f'{product_name}_private'],
                    alpha=alpha_fill,
                    color='red')

    # =========================
    # pie chart

    ax = axes[0][1]

    pie_labels = ['used', 'new']
    pie_values = [df_yesterday_mean[f'{product_name}_used'], df_yesterday_mean[f'{product_name}_new']]
    pie_explode = (0.01, 0.01)

    rgb_blue_fill = (12 / 255, 120 / 255, 237 / 255, 0.2)
    rgb_blue_edge = (12 / 255, 120 / 255, 237 / 255, 0.87)
    rgb_black_fill = (0, 0, 0, 0.2)
    rgb_black_edge = (0, 0, 0, 0.87)

    ax.pie(x=pie_values,
           labels=pie_labels,
           autopct='%1.1f%%',
           explode=pie_explode,
           textprops={'fontsize': 24},
           wedgeprops={'joinstyle': 'bevel', 'width': 0.2},
           colors=[rgb_black_fill, rgb_blue_fill])

    ax.set_aspect('equal')

    plt_img = plt.imread(fname=f'./icons/{product_name}.png', format='png')
    img_offset = OffsetImage(plt_img, zoom=1.2)
    img_annotation = AnnotationBbox(offsetbox=img_offset,
                                    xy=(0, 0),
                                    frameon=False)
    ax.add_artist(img_annotation)

    # =========================

    fig.suptitle(f"\n{product_name.replace('v', 'u').upper()}\n", fontsize=88)
    fig.tight_layout()

    avto_ru_doc = io.BytesIO()
    fig.savefig(avto_ru_doc, format='pdf')
    avto_ru_doc.seek(0)
    avto_ru_doc.name = f"{product_name.replace('v', 'u').upper()}.jpg"

    pdf_avto_ru = PdfReader(avto_ru_doc)

    return pdf_avto_ru


def get_avito_avto(df_result: 'DataFrame') -> 'pdf bytes':
    # prepare data to PLOT
    before_yesterday_val = datetime.date(datetime.today()) - timedelta(days=2)
    yesterday_val = datetime.date(datetime.today()) - timedelta(days=1)
    today_val = datetime.date(datetime.today())

    df_before_yesterday = df_result.query('datetime_of_day.dt.date < @yesterday_val')
    df_yesterday = df_result.query('datetime_of_day.dt.date >= @yesterday_val')

    # zeros to NaN
    df_before_yesterday = df_before_yesterday.mask(df_before_yesterday == 0)
    df_yesterday = df_yesterday.mask(df_yesterday == 0)

    df_before_yesterday_init = df_before_yesterday.copy(deep=True)
    df_yesterday_init = df_yesterday.copy(deep=True)

    # get mean for legend
    df_before_yesterday_mean = df_before_yesterday.mean(numeric_only=True)
    df_yesterday_mean = df_yesterday.mean(numeric_only=True)

    # fillna
    df_before_yesterday = df_before_yesterday.fillna(df_before_yesterday.median(numeric_only=True))
    df_yesterday = df_yesterday.fillna(df_yesterday.median(numeric_only=True))

    sns.set_theme()
    fig, axes = plt.subplots(nrows=3,
                             ncols=2,
                             figsize=(23, 32),
                             facecolor='white')

    label_config = {'fontsize': '16'}
    title_config = {'fontsize': '24'}

    product_name = 'avito_avto'

    alpha_fill = 0.1
    alpha_edge = 0.87
    alpha_edge_shadow = 0.2

    avito_purple = (129 / 255, 63 / 255, 229 / 255)
    avito_blue = (18 / 255, 151 / 255, 255 / 255)
    avito_red = (252 / 255, 37 / 255, 66 / 255)
    avito_green = (29 / 255, 222 / 255, 79 / 255)

    x_ticks_labels = pd.date_range(start=before_yesterday_val, end=yesterday_val, freq='5T')[:-1]
    x_ticks_labels = list(map(lambda x: x.strftime('%H:%M'), x_ticks_labels))[::24] + list(
        x_ticks_labels[-1:].strftime('%H:%M'))

    # create ticks and add last element for plot
    x_ticks = np.arange(df_yesterday.shape[0])
    x_ticks = x_ticks[::24]
    x_ticks = np.insert(x_ticks, len(x_ticks), df_yesterday.shape[0])

    # =================================
    # Total cars
    # =================================

    legend_mean_val = int(df_yesterday_mean[f'{product_name}_total'])
    ax = sns.lineplot(ax=axes[0][0],
                      x=np.arange(df_yesterday.shape[0]),
                      y=df_yesterday[f'{product_name}_total'],
                      alpha=alpha_edge,
                      color=avito_purple,
                      label=f'yesterday ({legend_mean_val} cars)')

    # ===============================
    # draw the points with 'no_cars'
    p_value, idx_arr_bfy, idx_arr_y = is_diff(arr_1=df_before_yesterday_init[f'{product_name}_total'].values,
                                              arr_2=df_yesterday_init[f'{product_name}_total'].values)

    ax.scatter(idx_arr_bfy,
               df_before_yesterday[f'{product_name}_total'].values[idx_arr_bfy],
               marker='o',
               s=50,
               color='black',
               alpha=alpha_edge)

    ax.scatter(idx_arr_y,
               df_yesterday[f'{product_name}_total'].values[idx_arr_y],
               marker='o',
               color=avito_purple,
               alpha=alpha_edge,
               s=50)
    # ==============================

    legend_mean_val = int(df_before_yesterday_mean[f'{product_name}_total'])
    ax.plot(np.arange(df_before_yesterday.shape[0]),
            df_before_yesterday[f'{product_name}_total'],
            alpha=alpha_edge_shadow,
            color='black',
            label=f'yesterday - 1 ({legend_mean_val} cars)')

    ax.set_xlabel('Time', **label_config)
    ax.set_ylabel('Total cars', **label_config)
    ax.set_title(f'Total cars\np_value = {round(p_value, 4)}', **title_config)

    ax.set_xticks(ticks=x_ticks,
                  labels=x_ticks_labels,
                  rotation=45)
    ax.margins(x=0.01)

    y_lims = ax.get_ylim()

    ax.fill_between(x=df_yesterday['time_of_day'],
                    y1=y_lims[0],
                    y2=df_yesterday[f'{product_name}_total'],
                    alpha=alpha_fill,
                    color=avito_purple)

    ax.legend(fontsize=18)

    # =================================
    # Used cars
    # =================================

    legend_mean_val = int(df_yesterday_mean[f'{product_name}_used'])
    ax = sns.lineplot(ax=axes[1][0],
                      x=df_yesterday['time_of_day'],
                      y=df_yesterday[f'{product_name}_used'],
                      alpha=alpha_edge,
                      color=avito_blue,
                      ls='--',
                      label=f'yesterday ({legend_mean_val})')

    legend_mean_val = int(df_before_yesterday_mean[f'{product_name}_used'])
    ax.plot(np.arange(df_before_yesterday.shape[0]),
            df_before_yesterday[f'{product_name}_used'],
            alpha=alpha_edge_shadow,
            color='black',
            label=f'yesterday - 1 ({legend_mean_val} cars)')

    # ===============================
    # draw the points with 'no_cars'
    p_value, idx_arr_bfy, idx_arr_y = is_diff(arr_1=df_before_yesterday_init[f'{product_name}_used'].values,
                                              arr_2=df_yesterday_init[f'{product_name}_used'].values)

    ax.scatter(idx_arr_bfy,
               df_before_yesterday[f'{product_name}_used'].values[idx_arr_bfy],
               marker='o',
               s=50,
               color='black',
               alpha=alpha_edge)

    ax.scatter(idx_arr_y,
               df_yesterday[f'{product_name}_used'].values[idx_arr_y],
               marker='o',
               color=avito_blue,
               alpha=alpha_edge,
               s=50)
    # ==============================

    ax.set_xlabel('Time', **label_config)
    ax.set_ylabel('Used cars', **label_config)
    ax.set_title(f'Used cars\np_value = {round(p_value, 4)}', **title_config)

    ax.set_xticks(ticks=x_ticks,
                  labels=x_ticks_labels,
                  rotation=45)
    ax.margins(x=0.01)

    ax.legend(fontsize=18)

    y_lims = ax.get_ylim()
    ax.fill_between(x=df_yesterday['time_of_day'],
                    y1=y_lims[0],
                    y2=df_yesterday[f'{product_name}_used'],
                    alpha=alpha_fill,
                    color=avito_blue)

    # =================================
    # New cars
    # =================================

    legend_mean_val = int(df_yesterday_mean[f'{product_name}_new'])
    ax = sns.lineplot(ax=axes[1][1],
                      x=df_yesterday['time_of_day'],
                      y=df_yesterday[f'{product_name}_new'],
                      alpha=alpha_edge,
                      color=avito_blue,
                      ls='--',
                      label=f'yesterday ({legend_mean_val})')

    legend_mean_val = int(df_before_yesterday_mean[f'{product_name}_new'])
    ax.plot(np.arange(df_before_yesterday.shape[0]),
            df_before_yesterday[f'{product_name}_new'],
            alpha=alpha_edge_shadow,
            color='black',
            label=f'yesterday - 1 ({legend_mean_val} cars)')

    # ===============================
    # draw the points with 'no_cars'
    p_value, idx_arr_bfy, idx_arr_y = is_diff(arr_1=df_before_yesterday_init[f'{product_name}_new'].values,
                                              arr_2=df_yesterday_init[f'{product_name}_new'].values)

    ax.scatter(idx_arr_bfy,
               df_before_yesterday[f'{product_name}_new'].values[idx_arr_bfy],
               marker='o',
               s=50,
               color='black',
               alpha=alpha_edge)

    ax.scatter(idx_arr_y,
               df_yesterday[f'{product_name}_new'].values[idx_arr_y],
               marker='o',
               color=avito_blue,
               alpha=alpha_edge,
               s=50)
    # ==============================

    ax.set_xlabel('Time', **label_config)
    ax.set_ylabel('New cars', **label_config)
    ax.set_title(f'New cars\np_value = {round(p_value, 4)}', **title_config)

    ax.set_xticks(ticks=x_ticks,
                  labels=x_ticks_labels,
                  rotation=45)
    ax.margins(x=0.01)

    ax.legend(fontsize=18)

    y_lims = ax.get_ylim()
    ax.fill_between(x=df_yesterday['time_of_day'],
                    y1=y_lims[0],
                    y2=df_yesterday[f'{product_name}_new'],
                    alpha=alpha_fill,
                    color=avito_blue)

    # =================================
    # Company cars
    # =================================

    legend_mean_val = int(df_yesterday_mean[f'{product_name}_company'])
    ax = sns.lineplot(ax=axes[2][0],
                      x=df_yesterday['time_of_day'],
                      y=df_yesterday[f'{product_name}_company'],
                      alpha=alpha_edge,
                      color=avito_green,
                      ls='-.',
                      label=f'yesterday ({legend_mean_val})')

    legend_mean_val = int(df_before_yesterday_mean[f'{product_name}_company'])
    ax.plot(np.arange(df_before_yesterday.shape[0]),
            df_before_yesterday[f'{product_name}_company'],
            alpha=alpha_edge_shadow,
            color='black',
            label=f'yesterday - 1 ({legend_mean_val} cars)')

    # ===============================
    # draw the points with 'no_cars'
    p_value, idx_arr_bfy, idx_arr_y = is_diff(arr_1=df_before_yesterday_init[f'{product_name}_company'].values,
                                              arr_2=df_yesterday_init[f'{product_name}_company'].values)

    ax.scatter(idx_arr_bfy,
               df_before_yesterday[f'{product_name}_company'].values[idx_arr_bfy],
               marker='o',
               s=50,
               color='black',
               alpha=alpha_edge)

    ax.scatter(idx_arr_y,
               df_yesterday[f'{product_name}_company'].values[idx_arr_y],
               marker='o',
               color=avito_green,
               alpha=alpha_edge,
               s=50)
    # ==============================

    ax.set_xlabel('Time', **label_config)
    ax.set_ylabel('Company cars', **label_config)
    ax.set_title(f'Company cars\np_value = {round(p_value, 4)}', **title_config)

    ax.set_xticks(ticks=x_ticks,
                  labels=x_ticks_labels,
                  rotation=45)
    ax.margins(x=0.01)

    ax.legend(fontsize=18)

    y_lims = ax.get_ylim()
    ax.fill_between(x=df_yesterday['time_of_day'],
                    y1=y_lims[0],
                    y2=df_yesterday[f'{product_name}_company'],
                    alpha=alpha_fill,
                    color=avito_green)

    # =================================
    # Private cars
    # =================================

    legend_mean_val = int(df_yesterday_mean[f'{product_name}_private'])
    ax = sns.lineplot(ax=axes[2][1],
                      x=df_yesterday['time_of_day'],
                      y=df_yesterday[f'{product_name}_private'],
                      alpha=alpha_edge,
                      color=avito_green,
                      ls='-.',
                      label=f'yesterday ({legend_mean_val})')

    legend_mean_val = int(df_before_yesterday_mean[f'{product_name}_private'])
    ax.plot(np.arange(df_before_yesterday.shape[0]),
            df_before_yesterday[f'{product_name}_private'],
            alpha=alpha_edge_shadow,
            color='black',
            label=f'yesterday - 1 ({legend_mean_val} cars)')

    # ===============================
    # draw the points with 'no_cars'
    p_value, idx_arr_bfy, idx_arr_y = is_diff(arr_1=df_before_yesterday_init[f'{product_name}_private'].values,
                                              arr_2=df_yesterday_init[f'{product_name}_private'].values)

    ax.scatter(idx_arr_bfy,
               df_before_yesterday[f'{product_name}_private'].values[idx_arr_bfy],
               marker='o',
               s=50,
               color='black',
               alpha=alpha_edge)

    ax.scatter(idx_arr_y,
               df_yesterday[f'{product_name}_private'].values[idx_arr_y],
               marker='o',
               color=avito_green,
               alpha=alpha_edge,
               s=50)
    # ==============================

    ax.set_xlabel('Time', **label_config)
    ax.set_ylabel('Private cars', **label_config)
    ax.set_title(f'Private cars\np_value = {round(p_value, 4)}', **title_config)

    ax.set_xticks(ticks=x_ticks,
                  labels=x_ticks_labels,
                  rotation=45)
    ax.margins(x=0.01)

    ax.legend(fontsize=18)

    y_lims = ax.get_ylim()
    ax.fill_between(x=df_yesterday['time_of_day'],
                    y1=y_lims[0],
                    y2=df_yesterday[f'{product_name}_private'],
                    alpha=alpha_fill,
                    color=avito_green)

    # =========================
    # pie chart

    ax = axes[0][1]

    pie_labels = ['used', 'new']
    pie_values = [df_yesterday_mean[f'{product_name}_used'], df_yesterday_mean[f'{product_name}_new']]
    pie_explode = (0.01, 0.01)

    rgb_blue_fill = (12 / 255, 120 / 255, 237 / 255, 0.2)
    rgb_blue_edge = (12 / 255, 120 / 255, 237 / 255, 0.87)
    rgb_black_fill = (0, 0, 0, 0.2)
    rgb_black_edge = (0, 0, 0, 0.87)

    ax.pie(x=pie_values,
           labels=pie_labels,
           autopct='%1.1f%%',
           explode=pie_explode,
           textprops={'fontsize': 24},
           wedgeprops={'joinstyle': 'bevel', 'width': 0.2},
           colors=[rgb_black_fill, avito_purple + (0.5,)])

    ax.set_aspect('equal')

    plt_img = plt.imread(fname=f'./icons/{product_name}.png', format='png')
    img_offset = OffsetImage(plt_img, zoom=1.2)
    img_annotation = AnnotationBbox(offsetbox=img_offset,
                                    xy=(0, 0),
                                    frameon=False)
    ax.add_artist(img_annotation)

    # =========================

    fig.suptitle(f'\n{product_name.upper()}\n', fontsize=88)
    fig.tight_layout()

    avito_avto_doc = io.BytesIO()
    fig.savefig(avito_avto_doc, format='pdf')
    avito_avto_doc.seek(0)
    avito_avto_doc.name = f'{product_name.upper()}.jpg'

    pdf_avito_avto = PdfReader(avito_avto_doc)

    return pdf_avito_avto


def get_drom(df_result: 'DataFrame') -> 'pdf bytes':
    # prepare data to PLOT
    before_yesterday_val = datetime.date(datetime.today()) - timedelta(days=2)
    yesterday_val = datetime.date(datetime.today()) - timedelta(days=1)
    today_val = datetime.date(datetime.today())

    df_before_yesterday = df_result.query('datetime_of_day.dt.date < @yesterday_val')
    df_yesterday = df_result.query('datetime_of_day.dt.date >= @yesterday_val')

    # zeros to NaN
    df_before_yesterday = df_before_yesterday.mask(df_before_yesterday == 0)
    df_yesterday = df_yesterday.mask(df_yesterday == 0)

    df_before_yesterday_init = df_before_yesterday.copy(deep=True)
    df_yesterday_init = df_yesterday.copy(deep=True)

    # get mean for legend
    df_before_yesterday_mean = df_before_yesterday.mean(numeric_only=True)
    df_yesterday_mean = df_yesterday.mean(numeric_only=True)

    # fillna
    df_before_yesterday = df_before_yesterday.fillna(df_before_yesterday.median(numeric_only=True))
    df_yesterday = df_yesterday.fillna(df_yesterday.median(numeric_only=True))

    sns.set_theme()
    fig, axes = plt.subplots(nrows=3,
                             ncols=2,
                             figsize=(23, 32),
                             facecolor='white')

    label_config = {'fontsize': '16'}
    title_config = {'fontsize': '24'}

    product_name = 'drom'

    alpha_fill = 0.1
    alpha_edge = 0.87
    alpha_edge_shadow = 0.2

    drom_red = (219 / 255, 0 / 255, 27 / 255)
    drom_black = (0, 0, 0)

    x_ticks_labels = pd.date_range(start=before_yesterday_val, end=yesterday_val, freq='5T')[:-1]
    x_ticks_labels = list(map(lambda x: x.strftime('%H:%M'), x_ticks_labels))[::24] + list(
        x_ticks_labels[-1:].strftime('%H:%M'))

    # create ticks and add last element for plot
    x_ticks = np.arange(df_yesterday.shape[0])
    x_ticks = x_ticks[::24]
    x_ticks = np.insert(x_ticks, len(x_ticks), df_yesterday.shape[0])

    # =================================
    # Total cars
    # =================================

    legend_mean_val = int(df_yesterday_mean[f'{product_name}_total'])
    ax = sns.lineplot(ax=axes[0][0],
                      x=np.arange(df_yesterday.shape[0]),
                      y=df_yesterday[f'{product_name}_total'],
                      alpha=alpha_edge,
                      color=drom_red,
                      label=f'yesterday ({legend_mean_val} cars)')

    # ===============================
    # draw the points with 'no_cars'
    p_value, idx_arr_bfy, idx_arr_y = is_diff(arr_1=df_before_yesterday_init[f'{product_name}_total'].values,
                                              arr_2=df_yesterday_init[f'{product_name}_total'].values)

    ax.scatter(idx_arr_bfy,
               df_before_yesterday[f'{product_name}_total'].values[idx_arr_bfy],
               marker='o',
               s=50,
               color='black',
               alpha=alpha_edge)

    ax.scatter(idx_arr_y,
               df_yesterday[f'{product_name}_total'].values[idx_arr_y],
               marker='o',
               color=drom_red,
               alpha=alpha_edge,
               s=50)
    # ==============================

    legend_mean_val = int(df_before_yesterday_mean[f'{product_name}_total'])
    ax.plot(np.arange(df_before_yesterday.shape[0]),
            df_before_yesterday[f'{product_name}_total'],
            alpha=alpha_edge_shadow,
            color='black',
            label=f'yesterday - 1 ({legend_mean_val} cars)')

    ax.set_xlabel('Time', **label_config)
    ax.set_ylabel('Total cars', **label_config)
    ax.set_title(f'Total cars\np_value = {round(p_value, 4)}', **title_config)

    ax.set_xticks(ticks=x_ticks,
                  labels=x_ticks_labels,
                  rotation=45)
    ax.margins(x=0.01)

    y_lims = ax.get_ylim()

    ax.fill_between(x=df_yesterday['time_of_day'],
                    y1=y_lims[0],
                    y2=df_yesterday[f'{product_name}_total'],
                    alpha=alpha_fill,
                    color=drom_red)

    ax.legend(fontsize=18)

    # =================================
    # Used cars
    # =================================

    legend_mean_val = int(df_yesterday_mean[f'{product_name}_used'])
    ax = sns.lineplot(ax=axes[1][0],
                      x=df_yesterday['time_of_day'],
                      y=df_yesterday[f'{product_name}_used'],
                      alpha=alpha_edge,
                      color=drom_black,
                      ls='--',
                      label=f'yesterday ({legend_mean_val})')

    legend_mean_val = int(df_before_yesterday_mean[f'{product_name}_used'])
    ax.plot(np.arange(df_before_yesterday.shape[0]),
            df_before_yesterday[f'{product_name}_used'],
            alpha=alpha_edge_shadow,
            color='black',
            label=f'yesterday - 1 ({legend_mean_val} cars)')

    # ===============================
    # draw the points with 'no_cars'
    p_value, idx_arr_bfy, idx_arr_y = is_diff(arr_1=df_before_yesterday_init[f'{product_name}_used'].values,
                                              arr_2=df_yesterday_init[f'{product_name}_used'].values)

    ax.scatter(idx_arr_bfy,
               df_before_yesterday[f'{product_name}_used'].values[idx_arr_bfy],
               marker='o',
               s=50,
               color='black',
               alpha=alpha_edge)

    ax.scatter(idx_arr_y,
               df_yesterday[f'{product_name}_used'].values[idx_arr_y],
               marker='o',
               color=drom_black,
               alpha=alpha_edge,
               s=70)
    # ==============================

    ax.set_xlabel('Time', **label_config)
    ax.set_ylabel('Used cars', **label_config)
    ax.set_title(f'Used cars\np_value = {round(p_value, 4)}', **title_config)

    ax.set_xticks(ticks=x_ticks,
                  labels=x_ticks_labels,
                  rotation=45)
    ax.margins(x=0.01)

    ax.legend(fontsize=18)

    y_lims = ax.get_ylim()
    ax.fill_between(x=df_yesterday['time_of_day'],
                    y1=y_lims[0],
                    y2=df_yesterday[f'{product_name}_used'],
                    alpha=alpha_fill,
                    color=drom_black)

    # =================================
    # New cars
    # =================================

    legend_mean_val = int(df_yesterday_mean[f'{product_name}_new'])
    ax = sns.lineplot(ax=axes[1][1],
                      x=df_yesterday['time_of_day'],
                      y=df_yesterday[f'{product_name}_new'],
                      alpha=alpha_edge,
                      color=drom_black,
                      ls='--',
                      label=f'yesterday ({legend_mean_val})')

    legend_mean_val = int(df_before_yesterday_mean[f'{product_name}_new'])
    ax.plot(np.arange(df_before_yesterday.shape[0]),
            df_before_yesterday[f'{product_name}_new'],
            alpha=alpha_edge_shadow,
            color='black',
            label=f'yesterday - 1 ({legend_mean_val} cars)')

    # ===============================
    # draw the points with 'no_cars'
    p_value, idx_arr_bfy, idx_arr_y = is_diff(arr_1=df_before_yesterday_init[f'{product_name}_new'].values,
                                              arr_2=df_yesterday_init[f'{product_name}_new'].values)

    ax.scatter(idx_arr_bfy,
               df_before_yesterday[f'{product_name}_new'].values[idx_arr_bfy],
               marker='o',
               s=50,
               color='black',
               alpha=alpha_edge)

    ax.scatter(idx_arr_y,
               df_yesterday[f'{product_name}_new'].values[idx_arr_y],
               marker='o',
               color=drom_black,
               alpha=alpha_edge,
               s=70)
    # ==============================

    ax.set_xlabel('Time', **label_config)
    ax.set_ylabel('New cars', **label_config)
    ax.set_title(f'New cars\np_value = {round(p_value, 4)}', **title_config)

    ax.set_xticks(ticks=x_ticks,
                  labels=x_ticks_labels,
                  rotation=45)
    ax.margins(x=0.01)

    ax.legend(fontsize=18)

    y_lims = ax.get_ylim()
    ax.fill_between(x=df_yesterday['time_of_day'],
                    y1=y_lims[0],
                    y2=df_yesterday[f'{product_name}_new'],
                    alpha=alpha_fill,
                    color=drom_black)

    # =================================
    # Company cars
    # =================================

    legend_mean_val = int(df_yesterday_mean[f'{product_name}_company'])
    ax = sns.lineplot(ax=axes[2][0],
                      x=df_yesterday['time_of_day'],
                      y=df_yesterday[f'{product_name}_company'],
                      alpha=alpha_edge,
                      color=drom_red,
                      ls='-.',
                      label=f'yesterday ({legend_mean_val})')

    legend_mean_val = int(df_before_yesterday_mean[f'{product_name}_company'])
    ax.plot(np.arange(df_before_yesterday.shape[0]),
            df_before_yesterday[f'{product_name}_company'],
            alpha=alpha_edge_shadow,
            color='black',
            label=f'yesterday - 1 ({legend_mean_val} cars)')

    # ===============================
    # draw the points with 'no_cars'
    p_value, idx_arr_bfy, idx_arr_y = is_diff(arr_1=df_before_yesterday_init[f'{product_name}_company'].values,
                                              arr_2=df_yesterday_init[f'{product_name}_company'].values)

    ax.scatter(idx_arr_bfy,
               df_before_yesterday[f'{product_name}_company'].values[idx_arr_bfy],
               marker='o',
               s=50,
               color='black',
               alpha=alpha_edge)

    ax.scatter(idx_arr_y,
               df_yesterday[f'{product_name}_company'].values[idx_arr_y],
               marker='o',
               color=drom_red,
               alpha=alpha_edge,
               s=50)
    # ==============================

    ax.set_xlabel('Time', **label_config)
    ax.set_ylabel('Company cars', **label_config)
    ax.set_title(f'Company cars\np_value = {round(p_value, 4)}', **title_config)

    # x_ticks = pd.concat([df_yesterday['time_of_day'].iloc[::24], df_yesterday['time_of_day'][-1:]])

    ax.set_xticks(ticks=x_ticks,
                  labels=x_ticks_labels,
                  rotation=45)
    ax.margins(x=0.01)

    ax.legend(fontsize=18)

    y_lims = ax.get_ylim()
    ax.fill_between(x=df_yesterday['time_of_day'],
                    y1=y_lims[0],
                    y2=df_yesterday[f'{product_name}_company'],
                    alpha=alpha_fill,
                    color=drom_red)

    # =================================
    # Private cars
    # =================================

    legend_mean_val = int(df_yesterday_mean[f'{product_name}_private'])
    ax = sns.lineplot(ax=axes[2][1],
                      x=df_yesterday['time_of_day'],
                      y=df_yesterday[f'{product_name}_private'],
                      alpha=alpha_edge,
                      color=drom_red,
                      ls='-.',
                      label=f'yesterday ({legend_mean_val})')

    legend_mean_val = int(df_before_yesterday_mean[f'{product_name}_private'])
    ax.plot(np.arange(df_before_yesterday.shape[0]),
            df_before_yesterday[f'{product_name}_private'],
            alpha=alpha_edge_shadow,
            color='black',
            label=f'yesterday - 1 ({legend_mean_val} cars)')

    # ===============================
    # draw the points with 'no_cars'
    p_value, idx_arr_bfy, idx_arr_y = is_diff(arr_1=df_before_yesterday_init[f'{product_name}_private'].values,
                                              arr_2=df_yesterday_init[f'{product_name}_private'].values)

    ax.scatter(idx_arr_bfy,
               df_before_yesterday[f'{product_name}_private'].values[idx_arr_bfy],
               marker='o',
               s=50,
               color='black',
               alpha=alpha_edge)

    ax.scatter(idx_arr_y,
               df_yesterday[f'{product_name}_private'].values[idx_arr_y],
               marker='o',
               color=drom_red,
               alpha=alpha_edge,
               s=50)
    # ==============================

    ax.set_xlabel('Time', **label_config)
    ax.set_ylabel('Private cars', **label_config)
    ax.set_title(f'Private cars\np_value = {round(p_value, 4)}', **title_config)

    ax.set_xticks(ticks=x_ticks,
                  labels=x_ticks_labels,
                  rotation=45)
    ax.margins(x=0.01)

    ax.legend(fontsize=18)

    y_lims = ax.get_ylim()
    ax.fill_between(x=df_yesterday['time_of_day'],
                    y1=y_lims[0],
                    y2=df_yesterday[f'{product_name}_private'],
                    alpha=alpha_fill,
                    color=drom_red)

    # =========================
    # pie chart

    ax = axes[0][1]

    pie_labels = ['used', 'new']
    pie_values = [df_yesterday_mean[f'{product_name}_used'], df_yesterday_mean[f'{product_name}_new']]
    pie_explode = (0.01, 0.01)

    rgb_blue_fill = (12 / 255, 120 / 255, 237 / 255, 0.2)
    rgb_blue_edge = (12 / 255, 120 / 255, 237 / 255, 0.87)
    rgb_black_fill = (0, 0, 0, 0.4)
    rgb_black_edge = (0, 0, 0, 0.87)

    ax.pie(x=pie_values,
           labels=pie_labels,
           autopct='%1.1f%%',
           explode=pie_explode,
           textprops={'fontsize': 24},
           wedgeprops={'joinstyle': 'bevel', 'width': 0.2},
           colors=[rgb_black_fill, drom_red + (0.9,)])

    ax.set_aspect('equal')

    plt_img = plt.imread(fname=f'./icons/{product_name}.png', format='png')
    img_offset = OffsetImage(plt_img, zoom=1.2)
    img_annotation = AnnotationBbox(offsetbox=img_offset,
                                    xy=(0, 0),
                                    frameon=False)
    ax.add_artist(img_annotation)

    # =========================

    fig.suptitle(f'\n{product_name.upper()}\n', fontsize=88)
    fig.tight_layout()

    drom_doc = io.BytesIO()
    fig.savefig(drom_doc, format='pdf')
    drom_doc.seek(0)
    drom_doc.name = f'{product_name.upper()}.jpg'

    pdf_drom = PdfReader(drom_doc)

    return pdf_drom


def get_sber_avto(df_result: 'DataFrame') -> 'pdf bytes':
    # prepare data to PLOT
    before_yesterday_val = datetime.date(datetime.today()) - timedelta(days=2)
    yesterday_val = datetime.date(datetime.today()) - timedelta(days=1)
    today_val = datetime.date(datetime.today())

    df_before_yesterday = df_result.query('datetime_of_day.dt.date < @yesterday_val')
    df_yesterday = df_result.query('datetime_of_day.dt.date >= @yesterday_val')

    # zeros to NaN
    df_before_yesterday = df_before_yesterday.mask(df_before_yesterday == 0)
    df_yesterday = df_yesterday.mask(df_yesterday == 0)

    df_before_yesterday_init = df_before_yesterday.copy(deep=True)
    df_yesterday_init = df_yesterday.copy(deep=True)

    # get mean for legend
    df_before_yesterday_mean = df_before_yesterday.mean(numeric_only=True)
    df_yesterday_mean = df_yesterday.mean(numeric_only=True)

    # fillna
    df_before_yesterday = df_before_yesterday.fillna(df_before_yesterday.median(numeric_only=True))
    df_yesterday = df_yesterday.fillna(df_yesterday.median(numeric_only=True))

    sns.set_theme()
    fig, axes = plt.subplots(nrows=2,
                             ncols=2,
                             figsize=(23, 23),
                             facecolor='white')

    label_config = {'fontsize': '16'}
    title_config = {'fontsize': '24'}

    product_name = 'sber_avto'

    alpha_fill = 0.1
    alpha_edge = 0.87
    alpha_edge_shadow = 0.2

    sber_avto_blue = (0, 63 / 255, 143 / 255)
    sber_avto_green = (33 / 255, 160 / 255, 56 / 255)

    x_ticks_labels = pd.date_range(start=before_yesterday_val, end=yesterday_val, freq='5T')[:-1]
    x_ticks_labels = list(map(lambda x: x.strftime('%H:%M'), x_ticks_labels))[::24] + list(
        x_ticks_labels[-1:].strftime('%H:%M'))

    # create ticks and add last element for plot
    x_ticks = np.arange(df_yesterday.shape[0])
    x_ticks = x_ticks[::24]
    x_ticks = np.insert(x_ticks, len(x_ticks), df_yesterday.shape[0])

    # =================================
    # Total cars
    # =================================

    legend_mean_val = int(df_yesterday_mean[f'{product_name}_total'])
    ax = sns.lineplot(ax=axes[0][0],
                      x=np.arange(df_yesterday.shape[0]),
                      y=df_yesterday[f'{product_name}_total'],
                      alpha=alpha_edge,
                      color=sber_avto_blue,
                      label=f'yesterday ({legend_mean_val} cars)')

    # ===============================
    # draw the points with 'no_cars'
    p_value, idx_arr_bfy, idx_arr_y = is_diff(arr_1=df_before_yesterday_init[f'{product_name}_total'].values,
                                              arr_2=df_yesterday_init[f'{product_name}_total'].values)

    ax.scatter(idx_arr_bfy,
               df_before_yesterday[f'{product_name}_total'].values[idx_arr_bfy],
               marker='o',
               s=50,
               color='black',
               alpha=alpha_edge)

    ax.scatter(idx_arr_y,
               df_yesterday[f'{product_name}_total'].values[idx_arr_y],
               marker='o',
               color=sber_avto_blue,
               alpha=alpha_edge,
               s=50)
    # ==============================

    legend_mean_val = int(df_before_yesterday_mean[f'{product_name}_total'])
    ax.plot(np.arange(df_before_yesterday.shape[0]),
            df_before_yesterday[f'{product_name}_total'],
            alpha=alpha_edge_shadow,
            color='black',
            label=f'yesterday - 1 ({legend_mean_val} cars)')

    ax.set_xlabel('Time', **label_config)
    ax.set_ylabel('Total cars', **label_config)
    ax.set_title(f'Total cars\np_value = {round(p_value, 4)}', **title_config)

    ax.set_xticks(ticks=x_ticks,
                  labels=x_ticks_labels,
                  rotation=45)
    ax.margins(x=0.01)

    y_lims = ax.get_ylim()

    ax.fill_between(x=df_yesterday['time_of_day'],
                    y1=y_lims[0],
                    y2=df_yesterday[f'{product_name}_total'],
                    alpha=alpha_fill,
                    color=sber_avto_blue)

    ax.legend(fontsize=18)

    # =================================
    # Used cars
    # =================================

    legend_mean_val = int(df_yesterday_mean[f'{product_name}_used'])
    ax = sns.lineplot(ax=axes[1][0],
                      x=df_yesterday['time_of_day'],
                      y=df_yesterday[f'{product_name}_used'],
                      alpha=alpha_edge,
                      color=sber_avto_green,
                      ls='--',
                      label=f'yesterday ({legend_mean_val})')

    legend_mean_val = int(df_before_yesterday_mean[f'{product_name}_used'])
    ax.plot(np.arange(df_before_yesterday.shape[0]),
            df_before_yesterday[f'{product_name}_used'],
            alpha=alpha_edge_shadow,
            color='black',
            label=f'yesterday - 1 ({legend_mean_val} cars)')

    # ===============================
    # draw the points with 'no_cars'
    p_value, idx_arr_bfy, idx_arr_y = is_diff(arr_1=df_before_yesterday_init[f'{product_name}_used'].values,
                                              arr_2=df_yesterday_init[f'{product_name}_used'].values)

    ax.scatter(idx_arr_bfy,
               df_before_yesterday[f'{product_name}_used'].values[idx_arr_bfy],
               marker='o',
               s=50,
               color='black',
               alpha=alpha_edge)

    ax.scatter(idx_arr_y,
               df_yesterday[f'{product_name}_used'].values[idx_arr_y],
               marker='o',
               color=sber_avto_green,
               alpha=alpha_edge,
               s=70)
    # ==============================

    ax.set_xlabel('Time', **label_config)
    ax.set_ylabel('Used cars', **label_config)
    ax.set_title(f'Used cars\np_value = {round(p_value, 4)}', **title_config)

    ax.set_xticks(ticks=x_ticks,
                  labels=x_ticks_labels,
                  rotation=45)
    ax.margins(x=0.01)

    ax.legend(fontsize=18)

    y_lims = ax.get_ylim()
    ax.fill_between(x=df_yesterday['time_of_day'],
                    y1=y_lims[0],
                    y2=df_yesterday[f'{product_name}_used'],
                    alpha=alpha_fill,
                    color=sber_avto_green)

    # =================================
    # New cars
    # =================================

    legend_mean_val = int(df_yesterday_mean[f'{product_name}_new'])
    ax = sns.lineplot(ax=axes[1][1],
                      x=df_yesterday['time_of_day'],
                      y=df_yesterday[f'{product_name}_new'],
                      alpha=alpha_edge,
                      color=sber_avto_green,
                      ls='--',
                      label=f'yesterday ({legend_mean_val})')

    legend_mean_val = int(df_before_yesterday_mean[f'{product_name}_new'])
    ax.plot(np.arange(df_before_yesterday.shape[0]),
            df_before_yesterday[f'{product_name}_new'],
            alpha=alpha_edge_shadow,
            color='black',
            label=f'yesterday - 1 ({legend_mean_val} cars)')

    # ===============================
    # draw the points with 'no_cars'
    p_value, idx_arr_bfy, idx_arr_y = is_diff(arr_1=df_before_yesterday_init[f'{product_name}_new'].values,
                                              arr_2=df_yesterday_init[f'{product_name}_new'].values)

    ax.scatter(idx_arr_bfy,
               df_before_yesterday[f'{product_name}_new'].values[idx_arr_bfy],
               marker='o',
               s=50,
               color='black',
               alpha=alpha_edge)

    ax.scatter(idx_arr_y,
               df_yesterday[f'{product_name}_new'].values[idx_arr_y],
               marker='o',
               color=sber_avto_green,
               alpha=alpha_edge,
               s=70)
    # ==============================

    ax.set_xlabel('Time', **label_config)
    ax.set_ylabel('New cars', **label_config)
    ax.set_title(f'New cars\np_value = {round(p_value, 4)}', **title_config)

    ax.set_xticks(ticks=x_ticks,
                  labels=x_ticks_labels,
                  rotation=45)
    ax.margins(x=0.01)

    ax.legend(fontsize=18)

    y_lims = ax.get_ylim()
    ax.fill_between(x=df_yesterday['time_of_day'],
                    y1=y_lims[0],
                    y2=df_yesterday[f'{product_name}_new'],
                    alpha=alpha_fill,
                    color=sber_avto_green)

    # =========================
    # pie chart
    # =========================

    ax = axes[0][1]

    pie_labels = ['used', 'new']
    pie_values = [df_yesterday_mean[f'{product_name}_used'], df_yesterday_mean[f'{product_name}_new']]
    pie_explode = (0.01, 0.05)

    rgb_blue_fill = (12 / 255, 120 / 255, 237 / 255, 0.2)
    rgb_blue_edge = (12 / 255, 120 / 255, 237 / 255, 0.87)
    rgb_black_fill = (0, 0, 0, 0.4)
    rgb_black_edge = (0, 0, 0, 0.87)

    ax.pie(x=pie_values,
           labels=pie_labels,
           autopct='%1.1f%%',
           explode=pie_explode,
           textprops={'fontsize': 24},
           wedgeprops={'joinstyle': 'bevel', 'width': 0.2},
           colors=[sber_avto_blue + (1,), sber_avto_green + (1,)])

    ax.set_aspect('equal')

    plt_img = plt.imread(fname=f'./icons/{product_name}.png', format='png')
    img_offset = OffsetImage(plt_img, zoom=1.2)
    img_annotation = AnnotationBbox(offsetbox=img_offset,
                                    xy=(0, 0),
                                    frameon=False)
    ax.add_artist(img_annotation)

    # =========================

    fig.suptitle(f'\n{product_name.upper()}\n',
                 fontsize=88)
    fig.tight_layout()

    sber_avto_doc = io.BytesIO()
    fig.savefig(sber_avto_doc, format='pdf')
    sber_avto_doc.seek(0)
    sber_avto_doc.name = f'{product_name.upper()}.jpg'

    pdf_sber_avto = PdfReader(sber_avto_doc)

    return pdf_sber_avto


def get_products_comparision(df_result: 'DataFrame') -> 'pdf bytes':
    yesterday_val = datetime.date(datetime.today()) - timedelta(days=1)

    df_before_yesterday = df_result.query('datetime_of_day.dt.date < @yesterday_val')
    df_yesterday = df_result.query('datetime_of_day.dt.date >= @yesterday_val')

    # zeros to NaN
    df_before_yesterday = df_before_yesterday.mask(df_before_yesterday == 0)
    df_yesterday = df_yesterday.mask(df_yesterday == 0)

    # get mean for legend
    df_before_yesterday_mean = df_before_yesterday.mean(numeric_only=True)
    df_yesterday_mean = df_yesterday.mean(numeric_only=True)

    seaborn_fill = (234 / 255, 234 / 255, 242 / 255)
    yesterday_color = (47 / 255, 47 / 255, 47 / 255)
    before_yesterday_color = (47 / 255, 47 / 255, 47 / 255, 0.2)

    sns.set_theme()
    fig, axes = plt.subplots(nrows=3,
                             ncols=2,
                             figsize=(23, 32),
                             facecolor=seaborn_fill)

    fontsize_title = 22

    # ===============================================================================
    # PRODUCT COMPARISON
    # ===============================================================================

    cohorts_list = [['total', 'used'], ['new', 'company'], ['private', '']]

    product_names = ['avto_ru', 'avito_avto', 'drom', 'sber_avto']

    # new for axes[i_row][j_column]
    for i_row, sub_cohorts in enumerate(cohorts_list):
        for j_column, cohort in enumerate(sub_cohorts):

            if cohort == '':
                continue

            product_labels, df_cohorts = get_cohort(cohort=cohort,
                                                    df_y_mean=df_yesterday_mean,
                                                    df_bfy_mean=df_before_yesterday_mean)

            ax = df_cohorts.iloc[:, :-1].plot(x='product',
                                              kind='bar',
                                              ax=axes[i_row][j_column],
                                              width=0.5, color=[before_yesterday_color,
                                                                yesterday_color])
            ax.set_xticklabels(labels=product_labels, rotation=0, fontsize=18)

            ax.get_xticklabels()

            ax.set_xlabel('')
            ax.set_title(f'\n\n{cohort.upper()} cars compare', fontsize=fontsize_title)

            fill_alpha = 1

            sber_avto_green = (33 / 255, 160 / 255, 56 / 255, fill_alpha)
            drom_red = (219 / 255, 0 / 255, 27 / 255, fill_alpha)
            avito_purple = (129 / 255, 63 / 255, 229 / 255, fill_alpha)
            avto_ru_red = (255 / 255, 0 / 255, 0 / 255, fill_alpha)
            avto_ru_blue = (12 / 255, 120 / 255, 237 / 255, fill_alpha)

            gray_fill = (0, 0, 0, 0.1)

            labels_suffix = ['bfy', 'y']
            for i, container in enumerate(ax.containers):
                for j, artist in enumerate(container):
                    coords = artist.get_xy()

                    # set image instead of
                    if i == 1:
                        plt_img = plt.imread(fname=f'./icons/{product_names[j]}.png', format='png')
                        img_offset = OffsetImage(plt_img, zoom=0.3)
                        img_annotation = AnnotationBbox(offsetbox=img_offset,
                                                        xy=(coords[0], coords[1]),
                                                        frameon=True,
                                                        bboxprops=dict(boxstyle='round, rounding_size=0.6, pad=0.5',
                                                                       facecolor='white',
                                                                       edgecolor='white'))
                        ax.add_artist(img_annotation)

                    ax.text(x=coords[0] + artist.get_width() / 2,
                            y=(coords[1] + artist.get_height()) / 1.2,
                            s=f'{labels_suffix[i]} = {artist.get_height()}',
                            rotation=90,
                            bbox=dict(facecolor='white', boxstyle='round'),
                            horizontalalignment='center')

            ax.margins(y=0.1)

            # change x_tick color
            ax.xaxis.set_tick_params(which='major', direction='out', pad=12)
            for tick in ax.get_xticklabels():
                if tick.get_text()[0] == '+':
                    tick.set_color('green')
                else:
                    tick.set_color('red')

            ax.legend(fontsize=18)

    # ===========================================================================================================
    # ADD INFO

    ax = axes[2][1]
    ax.text(x=0.5,
            y=0.5,
            s='Saint-Petersburg\n\nGeo +0 km.',
            horizontalalignment='center',
            fontsize=48,
            bbox=dict(facecolor='white',
                      boxstyle='round, pad=2'))
    ax.axis('off')
    # ===========================================================================================================

    fig.suptitle('\nProducts comparison\n', fontsize=88)
    fig.tight_layout()

    compare_doc = io.BytesIO()
    fig.savefig(compare_doc, format='pdf')
    compare_doc.seek(0)
    compare_doc.name = 'PRODUCT_COMPARISON.pdf'

    pdf_compare_doc = PdfReader(compare_doc)

    return pdf_compare_doc


def get_df_result() -> 'DataFrame':
    with open('./sql_queries/last_day_query.txt', 'r') as file:
        last_day_query = file.read()

    last_2_days_result = list(zip(*get_query_result(query=last_day_query)))
    result_headers = ['id', 'unix_timestamp',
                      'avto_ru_total', 'avto_ru_used', 'avto_ru_company',
                      'avito_avto_total', 'avito_avto_used', 'avito_avto_company',
                      'drom_total', 'drom_used', 'drom_company',
                      'sber_avto_total', 'sber_avto_used']

    # create new dict with headers
    test_dict = dict(list(zip(result_headers, last_2_days_result)))
    df_result = pd.DataFrame(data=test_dict)

    # dummy column
    df_result['sber_avto_company'] = 0

    df_result['time_of_day'] = df_result['unix_timestamp'] \
        .apply(lambda x: datetime.fromtimestamp(int(x)).strftime('%H:%M'))

    # to_datetime set UTC +0 need +3 then use timedelta
    df_result['datetime_of_day'] = pd.to_datetime(df_result['unix_timestamp'], unit='s') + pd.Timedelta('03:00:00')

    product_names = ['avto_ru', 'avito_avto', 'drom', 'sber_avto']

    # add NEW and PRIVATE
    for name in product_names:
        temp_avto = df_result[[f'{name}_total', f'{name}_used', f'{name}_company']].values

        new_cars, private_cars = fill_no_cars(arr=temp_avto)

        df_result[f'{name}_new'] = new_cars
        df_result[f'{name}_private'] = private_cars

    return df_result


envs = dotenv_values('/home/server_bot/.env')
bot = telegram.Bot(token=envs['AVTOBOT_TELEGRAM_TOKEN'])

# prepare final document
report_io = io.BytesIO()
report_pdf = PdfWriter()

# get pdf pages
df_result = get_df_result()
pdf_compare_doc = get_products_comparision(df_result=df_result)
pdf_avto_ru = get_avto_ru(df_result=df_result)
pdf_avito_avto = get_avito_avto(df_result=df_result)
pdf_drom = get_drom(df_result=df_result)
pdf_sber_avto = get_sber_avto(df_result=df_result)

# add pages to final document
report_pdf.add_page(pdf_compare_doc.pages[0])
report_pdf.add_page(pdf_avto_ru.pages[0])
report_pdf.add_page(pdf_avito_avto.pages[0])
report_pdf.add_page(pdf_drom.pages[0])
report_pdf.add_page(pdf_sber_avto.pages[0])

report_filename = f"Products_comparison_{datetime.now().date().strftime(format='%d_%m_%Y')}.pdf"

report_pdf.write(report_io)
report_io.seek(0)
report_io.name = report_filename

# additional backup
with open(f'./report_backups/{report_filename}', 'wb') as file:
    file.write(report_io.getvalue())

bot.send_document(chat_id=envs['AVTOBOT_CHAT_ID'],
                  document=report_io)
