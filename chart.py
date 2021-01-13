import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import FR, MO, SA, SU, TH, TU, WE
from pylab import rcParams
import os
from datetime import datetime

DATA_FILE = 'vendor/dssg_data.csv'
SAMPLES_FILE = 'vendor/dssg_samples.csv'
VACCINES_FILE = 'vendor/owid_vaccines.csv'
META_FILE = 'data/regions.txt'
GROUPS_FILE = 'data/groups.txt'

REGION_COLUMNS = dict(
    arsnorte='Norte',
    arscentro='Centro',
    arslvt='Lisboa e Vale do Tejo',
    arsalentejo='Alentejo',
    arsalgarve='Algarve',
    acores='Açores',
    madeira='Madeira')

REGION_POP = dict(
    arsnorte=3682370,
    arscentro=1744525,
    arslvt=3659871,
    arsalentejo=509849,
    arsalgarve=451006,
    acores=242796,
    madeira=254254)

AGE_COLUMNS = {
    '0_9': '0-9',
    '10_19': '10-19',
    '20_29': '20-29',
    '30_39': '30-39',
    '40_49': '40-49',
    '50_59': '50-59',
    '60_69': '60-69',
    '70_79': '70-79',
    '80_plus': '80+'
}

TOTAL_POP = sum(REGION_POP.values())

COL_DATE = 'data_dados'
COL_TOTAL = 'confirmados_novos'
COL_REGION_CONFIRMED = ['confirmados_' + k for k in REGION_COLUMNS.keys()]
COL_REGION_DEATHS = ['obitos_' + k for k in REGION_COLUMNS.keys()]
COL_REGION_RECOVERED = ['recovered_' + k for k in REGION_COLUMNS.keys()]
COL_AGE = sum([[
        'confirmados_' + k + '_m',
        'confirmados_' + k + '_f',
        'obitos_' + k + '_m',
        'obitos_' + k + '_f',
    ] for k in AGE_COLUMNS.keys()], [])


DPI = 150
WIDTH = 1200
HEIGHT = 675
LAST_WEEKDAY = 'SUN'


def main():
    os.makedirs('output', exist_ok=True)

    setup()

    print('Loading data')
    data = load_data()
    print('Processing new cases')

    data = new(data, COL_REGION_CONFIRMED + COL_REGION_DEATHS + COL_AGE + ['obitos', 'n_confirmados', 'recuperados', 'internados', 'internados_uci'])

    print('Plotting charts')

    print('newcases.png')
    plot_confirmed(data)
    plt.savefig('output/newcases.png')

    print('newcases_90d.png')
    plot_confirmed(data, -90)
    plt.savefig('output/newcases_90d.png')

    print('newcases_noroll.png')
    plot_confirmed(data, rolling=False)
    plt.savefig('output/newcases_90d_noroll.png')

    print('newcases_90d_noroll.png')
    plot_confirmed(data, -90, rolling=False)
    plt.savefig('output/newcases_90d_noroll.png')

    print('newdeaths.png')
    plot_deaths(data)
    plt.savefig('output/newdeaths.png')

    print('newdeaths_90d.png')
    plot_deaths(data, -90)
    plt.savefig('output/newdeaths_90d.png')

    print('newdeaths_90d_noroll.png')
    plot_deaths(data, -90, rolling=False)
    plt.savefig('output/newdeaths_90d_noroll.png')

    print('national.png')
    plot_global(data)
    plt.savefig('output/national.png')

    print('newcases_percent.png')
    plot_confirmed_percent(data)
    plt.savefig('output/newcases_percent.png')

    print('newcases_percent_noroll.png')
    plot_confirmed_percent(data, rolling=False)
    plt.savefig('output/newcases_percent_noroll.png')

    print('newcases_age.png')
    plot_confirmed_ages(data)
    plt.savefig('output/newcases_age.png')

    print('newdeaths_age.png')
    plot_deaths_age(data)
    plt.savefig('output/newdeaths_age.png')

    print('hospital.png')
    plot_hospital(data)
    plt.savefig('output/hospital.png')

    print('active.png')
    plot_active(data)
    plt.savefig('output/active.png')

    print('tests.png')
    plot_tests(data)
    plt.savefig('output/tests.png')

    print('vaccines.png')
    plot_vaccines(data)
    plt.savefig('output/vaccines.png')

    print('age_heatmap_cases.png')
    plot_age_heatmap(data, mode='cases')
    plt.savefig('output/age_heatmap_cases.png')

    print('age_heatmap_deaths.png')
    plot_age_heatmap(data, mode='deaths')
    plt.savefig('output/age_heatmap_deaths.png')


    print('newdeaths_national.png')
    plot_deaths_national(data)
    plt.savefig('output/newdeaths_national.png')


def plot_confirmed(data, first_row=0, rolling=True):
    x = data[COL_DATE]
    fig, ax = plot_init()

    last_date = data[COL_DATE].iloc[-1]

    for col, label in REGION_COLUMNS.items():
        key = 'new_confirmados_' + col

        y = (data[key]
            .div(REGION_POP[col])
            .mul(100000))

        if rolling:
            y = y.rolling(7).mean()

        p = plt.plot(
            x[first_row:],
            y[first_row:],
            label=label,
            marker='o',
            markersize=1.5)

        plt.axhline(
            y=y.iloc[-1],
            color=p[0].get_color(),
            linestyle='solid',
            linewidth=1,
            alpha=0.7)

    y = (data[COL_TOTAL]
            .div(TOTAL_POP)
            .mul(100000))

    if rolling:
        y = y.rolling(7).mean()

    # whole country

    p = plt.plot(
        x[first_row:],
        y[first_row:],
        label='País',
        color='#000000',
        marker='o',
        markersize=1.5)

    plt.axhline(
        y=y.iloc[-1],
        color='#000000',
        linestyle='solid',
        linewidth=1,
        alpha=1)

    plt.legend(loc='upper left')

    title = r'$\bf{' + 'COVID19\\ Portugal' + '}$ | Novos positivos / 100.000 habitantes | '
    if rolling:
        title += 'Média móvel de 7 dias | '
    else:
        title += 'Sem média móvel | '

    title += last_date.strftime('%Y-%m-%d')
    plt.title(title, loc='left')

    plot_footer()


def plot_deaths(data, first_row=0, rolling=True):
    x = data[COL_DATE]

    fig, ax = plot_init()

    for col, label in REGION_COLUMNS.items():
        y = (data['new_obitos_' + col]
                .div(REGION_POP[col])
                .mul(100000))

        if rolling:
            y = y.rolling(7).mean()

        p = plt.plot(
            x[first_row:],
            y[first_row:],
            label=label,
            marker='o',
            markersize=1.5)

        plt.axhline(
            y=y.iloc[-1],
            color=p[0].get_color(),
            linestyle='solid',
            linewidth=1,
            alpha=0.7)

    y = (data['new_obitos']
            .div(TOTAL_POP)
            .mul(100000))

    if rolling:
        y = y.rolling(7).mean()

    p = plt.plot(
        x[first_row:],
        y[first_row:],
        label='País',
        color='#000000',
        marker='o',
        markersize=1.5)

    plt.axhline(
        y=y.iloc[-1],
        color='#000000',
        linestyle='solid',
        linewidth=1,
        alpha=1)

    plt.legend(loc='upper left')

    title = r'$\bf{' + 'COVID19\\ Portugal' + '}$ | Novos óbitos / 100.000 habitantes | '
    if rolling:
        title += 'Média móvel de 7 dias | '
    else:
        title += 'Sem média móvel | '
    title += data[COL_DATE].iloc[-1].strftime('%Y-%m-%d')
    plt.title(title, loc='left',)

    plot_footer()

def plot_global(data, first_row=0):
    x = data[COL_DATE]

    fig, ax1 = plot_init()

    ax1.axes.get_yaxis().set_visible(False)
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax4 = ax1.twinx()

    ax1.minorticks_on()
    ax2.minorticks_on()
    ax3.minorticks_on()
    ax4.minorticks_on()

    # Novos óbitos
    ax2.spines["right"].set_position(("outward", 10))
    ax2.spines["right"].set_color('#000000')
    deaths = ax2.plot(
        x[first_row:],
        data['obitos']
            .div(TOTAL_POP)
            .mul(100000)
            .rolling(7)
            .mean()[first_row:],
        label='Óbitos',
        color='#000000',
        marker='o',
        markersize=1.5)

    # Novos confirmados
    ax3.spines["right"].set_position(("outward", 50))
    ax3.spines["right"].set_color('#ff0000')
    confirmed = ax3.plot(
        x[first_row:],
        data['confirmados']
            .div(TOTAL_POP)
            .mul(100000)
            .rolling(7)
            .mean()[first_row:],
        label='Confirmados',
        color='#ff0000',
        marker='o',
        markersize=1.5)


    # Recuperados
    ax4.spines["right"].set_position(("outward", 90))
    ax4.spines["right"].set_color('#00ff00')
    confirmed = ax3.plot(
        x[first_row:],
        data['recuperados']
            .div(TOTAL_POP)
            .mul(100000)
            .rolling(7)
            .mean()[first_row:],
        label='Recuperados',
        color='#00ff00',
        marker='o',
        markersize=1.5)

    # legend
    lns = deaths + confirmed
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')

    title = r'$\bf{' + 'COVID19\\ Portugal' + '}$ | Média móvel de 7 dias | '
    title += data[COL_DATE].iloc[-1].strftime('%Y-%m-%d')
    plt.title(title, loc='left',)

    plot_footer()

def plot_combined(data):
    x = data[COL_DATE]

    fig, ax = plot_init()

    plt.plot(
        x,
        data['obitos']
            .div(TOTAL_POP)
            .mul(100000)
            .rolling(7)
            .mean(),
        label='Óbitos')

    plt.plot(
        x,
        data['confirmados_novos']
            .div(TOTAL_POP)
            .mul(100000)
            .rolling(7)
            .mean(),
        label='Novos casos')

    title = r'$\bf{' + 'COVID19\\ Portugal' + '}$ | Média móvel de 7 dias | '
    title += data[COL_DATE].iloc[-1].strftime('%Y-%m-%d')
    plt.title(title, loc='left')

    plot_footer()

def plot_confirmed_percent(data, first_row=0, rolling=True):

    last_date = data[COL_DATE].iloc[-1]

    x = data[COL_DATE]

    fig, ax = plot_init()

    y = ((data['confirmados_novos'] / data['amostras_novas'])
            .mul(100))

    if rolling:
        y = y.rolling(7).mean()

    p = plt.plot(
        x[first_row:],
        y[first_row:],
        color='#000000',
        marker='o',
        markersize=1.5,
        label='Testes positivos (%)')


    plt.axhline(
        y=y.iloc[-1],
        color='#000000',
        linestyle='solid',
        linewidth=1,
        alpha=1)

    title = r'$\bf{' + 'COVID19\\ Portugal' + '}$ | Percentagem de testes positivos | '
    if rolling:
        title += 'Média móvel de 7 dias | '
    else:
        title += 'Sem média móvel | '
    title += last_date.strftime('%Y-%m-%d')
    plt.title(title, loc='left')

    plt.legend(loc='upper left')

    plot_footer()

def plot_confirmed_ages(data):

    last_date = data[COL_DATE].iloc[-1]

    x = data[COL_DATE]

    fig, ax = plot_init()

    df = pd.DataFrame()
    labels = []

    # sum m and f, and add to dataframe
    for k, v in AGE_COLUMNS.items():
        df[k] = data['new_confirmados_' + k + '_f'] + data['new_confirmados_' + k + '_m']
        labels.append(v)


    # turn into percentages
    df = df.apply(lambda r: r/r.sum(), axis=1).rolling(7, min_periods=1).mean().mul(100)


    plt.stackplot(x, df.transpose(), labels=labels)
    

    ####

    # reverse legend
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(reversed(handles), reversed(labels), loc='upper left')

    title = r'$\bf{' + 'COVID19\\ Portugal' + '}$ | Novos positivos por faixa etária (%) | Média móvel de 7 dias | '
    title += last_date.strftime('%Y-%m-%d')
    plt.title(title, loc='left')

    plot_footer()

def plot_deaths_age(data):

    last_date = data[COL_DATE].iloc[-1]

    x = data[COL_DATE]

    fig, ax = plot_init()

    df = pd.DataFrame()
    labels = []

    # sum m and f, and add to dataframe
    for k, v in AGE_COLUMNS.items():
        df[k] = data['new_obitos_' + k + '_f'] + data['new_obitos_' + k + '_m']
        labels.append(v)


    # turn into percentages
    df = df.apply(lambda r: r/r.sum(), axis=1).rolling(7, min_periods=1).mean().mul(100)


    plt.stackplot(x, df.transpose(), labels=labels)
    

    ####

    # reverse legend
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(reversed(handles), reversed(labels), loc='upper left')

    title = r'$\bf{' + 'COVID19\\ Portugal' + '}$ | Novos óbitos por faixa etária (%) | Média móvel de 7 dias | '
    title += last_date.strftime('%Y-%m-%d')
    plt.title(title, loc='left')

    plot_footer()


def plot_hospital(data):

    last_date = data[COL_DATE].iloc[-1]

    x = data[COL_DATE]

    fig, ax = plot_init()

    y = data['internados_enfermaria']
    p = plt.plot(
        x,
        y,
        label='Enfermaria',
        color='#000000',
        marker='o',
        markersize=1.5)
    plt.axhline(
        y=y.iloc[-1],
        color=p[0].get_color(),
        linestyle='solid',
        linewidth=1,
        alpha=0.7)

    y = data['internados_uci']
    p = plt.plot(
        x,
        y,
        label='Cuidados intensivos',
        color='#DD0000',
        marker='o',
        markersize=1.5)
    plt.axhline(
        y=y.iloc[-1],
        color=p[0].get_color(),
        linestyle='solid',
        linewidth=1,
        alpha=0.7)

    ####

    plt.legend(loc='upper left')

    title = r'$\bf{' + 'COVID19\\ Portugal' + '}$ | Hospitalizações | '
    title += last_date.strftime('%Y-%m-%d')
    plt.title(title, loc='left')

    plot_footer()


def plot_active(data):

    last_date = data[COL_DATE].iloc[-1]

    x = data[COL_DATE]

    fig, ax = plot_init()

    y = data['ativos']

    p = plt.plot(
        x,
        y,
        color='#000000',
        label='Casos ativos',
        marker='o',
        markersize=1.5)

    title = r'$\bf{' + 'COVID19\\ Portugal' + '}$ | Casos ativos | '
    title += last_date.strftime('%Y-%m-%d')
    plt.title(title, loc='left')

    plt.legend(loc='upper left')

    plot_footer()


def plot_vaccines(data):
    fig, ax = plot_init(daily=True)

    data = data.dropna(subset=['total_vaccinations'])
    last_date = data[COL_DATE].iloc[-1]

    x = data[COL_DATE]
    y = data['total_vaccinations']

    p = plt.plot(
        x,
        y,
        color='#000000',
        label='Vacinas administradas',
        marker='o',
        markersize=1.5)

    title = r'$\bf{' + 'COVID19\\ Portugal' + '}$ | Vacinas (Our World in Data) | '
    title += last_date.strftime('%Y-%m-%d')
    plt.title(title, loc='left')

    plt.legend(loc='upper left')

    plot_footer(top=True)


def plot_age_heatmap(data, mode='cases'):
    last_date = data[COL_DATE].iloc[-1]
    by_week = data.groupby([pd.Grouper(key=COL_DATE, freq='W-' + LAST_WEEKDAY)]).sum()

    if mode == 'cases':
        column = 'new_confirmados'
        title = 'Novos casos confirmados'
    elif mode == 'deaths':
        column = 'new_obitos'
        title = 'Novos óbitos'
    else:
        return

    # extract date column
    matrix = []
    x_labels = []
    y_labels = list(AGE_COLUMNS.values())

    for i, row in by_week.iterrows():
        x_labels.append((i - pd.Timedelta(pd.offsets.Day(6))).strftime('%d/%m'))

        # days in week so far
        days_in_week = (last_date - (i - pd.Timedelta(pd.offsets.Day(7)))).days

        l = []
        for k, v in AGE_COLUMNS.items():
            v = row[column + '_' + k + '_f'] + row[column + '_' + k + '_m']

            if i == by_week.index[-1]:
                v = (v / days_in_week) * 7

            l.append(v)

        matrix.append(l)

    matrix = np.array(matrix).transpose()

    fig, ax = plot_init(nogrid=True)
    #ax = plt.gca()

    # plot heatmap
    im = ax.imshow(matrix, aspect='auto', origin='lower', cmap='turbo')

    # create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, aspect=50)
    cbar.ax.set_ylabel("", rotation=-90, va="bottom")

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90)

    # turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(np.arange(len(x_labels) + 1)-.5, minor=True)
    ax.set_yticks(np.arange(len(y_labels) + 1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # title
    title = r'$\bf{' + 'COVID19\\ Portugal' + '}$ | ' + title + ' por semana, por faixa etária (última semana ajustada)'
    plt.title(title, loc='left')

    plot_footer(zero_origin=False)


def plot_tests(data):

    last_date = data[COL_DATE].iloc[-1]

    x = data[COL_DATE]

    fig, ax1 = plot_init()

    y = data['amostras_novas'].rolling(7).mean()
    p1 = ax1.plot(
        x,
        y,
        label='Testes realizados',
        color='#000000',
        marker='o',
        markersize=1.5)

    ####

    title = r'$\bf{' + 'COVID19\\ Portugal' + '}$ | Testes por dia | Média móvel de 7 dias | '
    title += last_date.strftime('%Y-%m-%d')
    plt.title(title, loc='left')

    plt.legend(loc='upper left')

    plot_footer()


def plot_deaths_national(data):
    fig, ax = plot_init()

    last_date = data[COL_DATE].iloc[-1]

    x = data[COL_DATE]
    y = data['new_obitos']

    p = plt.plot(
        x,
        y,
        color='#000000',
        label='Óbitos diários',
        marker='o',
        markersize=1.5)

    title = r'$\bf{' + 'COVID19\\ Portugal' + '}$ | Óbitos por dia | '
    title += last_date.strftime('%Y-%m-%d')
    plt.title(title, loc='left')

    plt.legend(loc='upper left')

    plot_footer(top=True)

#####

def setup():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)

def plot_init(daily=False, nogrid=False):
    plt.clf()
    plt.style.use('default')
    rcParams["font.family"] = "Cantarell"
    rcParams['axes.xmargin'] = .02
    rcParams['axes.ymargin'] = .02
    rcParams['axes.titlesize'] = 'medium'
    fig, ax = plt.subplots(figsize=(WIDTH/DPI, HEIGHT/DPI), dpi=DPI, constrained_layout=True)

    if not daily:
        days = mdates.DayLocator()
        weeks = mdates.WeekdayLocator(byweekday=MO)
        week_fmt = mdates.DateFormatter('%d/%m')
        ax.xaxis.set_major_locator(weeks)
        ax.xaxis.set_major_formatter(week_fmt)
        ax.xaxis.set_minor_locator(days)
        ax.set_xlabel('dias (marca a cada segunda-feira)')
    else:
        days = mdates.DayLocator()
        ax.xaxis.set_major_locator(days)
        day_fmt = mdates.DateFormatter('%d/%m')
        ax.xaxis.set_major_formatter(day_fmt)

    plt.xticks(rotation=45, fontsize=8)
    if not nogrid:
        ax.grid(axis='both', color='#000000', alpha=0.05)


    return fig, ax


def plot_footer(top=False, zero_origin=True):
    if top:
        x, y = 0.985, 0.975
    else:
        x, y = 0.985, 0.023

    plt.figtext(x, y, 'https://covid19.tdias.pt', horizontalalignment='right', verticalalignment='center', color='#BBBBBB')

    if zero_origin:
        plt.gca().set_ylim(bottom=0)

#####


def load_data():
    data_main = pd.read_csv(DATA_FILE)
    samples = pd.read_csv(SAMPLES_FILE)
    vaccines = pd.read_csv(VACCINES_FILE)

    # change vaccines date format to DD-MM-YYYY (like DSSG)
    vaccines['date'] = pd.to_datetime(vaccines["date"], format='%Y-%m-%d').dt.strftime('%d-%m-%Y')

    data = pd.merge(data_main, samples, how='left', left_on='data', right_on='data')
    data = pd.merge(data, vaccines, how='left', left_on='data', right_on='date')

    data[COL_DATE] = pd.to_datetime(data_main[COL_DATE], format='%d-%m-%Y %H:%M')

    #print(data.iloc[-1])
    #exit()

    return data


def extract_confirmed(data):
    return data[[COL_DATE] + [COL_TOTAL] + ['confirmados_' + k for k in REGION_COLUMNS.keys()]].copy()


def new(data, columns):
    """
    Calculates per-day values for a list of columns.

    :param data: DataFrame
    :param columns: List of column keys
    :return: New DataFrame with new_key columns
    """

    data = data.copy()

    for col in columns:
        new = []
        prev = float('nan')
        for v in data[col]:
            new.append(v - prev)
            prev = v

        data['new_' + col] = new

    return data


def region_new(data, prefix):
    return new(data, [prefix + k for k in REGION_COLUMNS.keys()])

if __name__ == '__main__':
    main()
