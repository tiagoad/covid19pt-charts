import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import FR, MO, SA, SU, TH, TU, WE
from pylab import rcParams
import os

DATA_FILE = 'data/data.csv'
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
TOTAL_POP = sum(REGION_POP.values())

COL_DATE = 'data_dados'
COL_TOTAL = 'confirmados_novos'
COL_REGION_CONFIRMED = ['confirmados_' + k for k in REGION_COLUMNS.keys()]
COL_REGION_DEATHS = ['obitos_' + k for k in REGION_COLUMNS.keys()]
COL_REGION_RECOVERED = ['recovered_' + k for k in REGION_COLUMNS.keys()]

DPI = 150
WIDTH = 1200
HEIGHT = 675


def main():
    os.makedirs('output', exist_ok=True)

    print('Loading data')
    data = load_data()
    print('Processing new cases')
    data_new = new(data, COL_REGION_CONFIRMED + COL_REGION_DEATHS + ['obitos', 'n_confirmados', 'recuperados', 'internados', 'internados_uci'])

    print('Plotting charts')
    plot_confirmed(data_new)
    plt.savefig('output/newcases.png')

    plot_confirmed(data_new, -90)
    plt.savefig('output/newcases_90d.png')

    plot_deaths(data_new)
    plt.savefig('output/newdeaths.png')

    plot_deaths(data_new, -90)
    plt.savefig('output/newdeaths_90d.png')

    plot_global(data_new)
    plt.savefig('output/national.png')


def plot_confirmed(data, first_row=0):
    x = data[COL_DATE]

    fig, ax = plot_init()

    last_date = data[COL_DATE].iloc[-1]

    for col, label in REGION_COLUMNS.items():
        key = 'new_confirmados_' + col

        y = (data[key]
            .div(REGION_POP[col])
            .mul(100000)
            .rolling(7)
            .mean())

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
            .mul(100000)
            .rolling(7)
            .mean())

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

    title = r'$\bf{' + 'COVID19\\ Portugal' + '}$ | Novos casos / 100.000 habitantes | Média móvel de 7 dias | '
    title += last_date.strftime('%Y-%m-%d')
    plt.title(title, loc='left')

    plot_footer()


def plot_deaths(data, first_row=0):
    x = data[COL_DATE]

    fig, ax = plot_init()

    for col, label in REGION_COLUMNS.items():
        y = (data['new_obitos_' + col]
                .div(REGION_POP[col])
                .mul(100000)
                .rolling(7)
                .mean())

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
            .mul(100000)
            .rolling(7)
            .mean())

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

    title = r'$\bf{' + 'COVID19\\ Portugal' + '}$ | Novos óbitos / 100.000 habitantes | Média móvel de 7 dias | '
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


#####

def plot_init():
    plt.clf()
    plt.style.use('default')
    rcParams["font.family"] = "Cantarell"
    rcParams['axes.xmargin'] = .02
    rcParams['axes.ymargin'] = .02
    rcParams['axes.titlesize'] = 'medium'
    fig, ax = plt.subplots(figsize=(WIDTH/DPI, HEIGHT/DPI), dpi=DPI, constrained_layout=True)
    months = mdates.MonthLocator()
    days = mdates.DayLocator()
    weeks = mdates.WeekdayLocator(byweekday=MO)
    months_fmt = mdates.DateFormatter('%m/%Y')
    week_fmt = mdates.DateFormatter('%d/%m')
    ax.xaxis.set_major_locator(weeks)
    ax.xaxis.set_major_formatter(week_fmt)
    ax.xaxis.set_minor_locator(days)
    plt.xticks(rotation=45, fontsize=8)
    ax.grid(axis='both', color='#F0F0F0')
    ax.set_xlabel('semana (segunda-feira)')
    return fig, ax


def plot_footer():
    plt.figtext(0.985, 0.023, 'Fonte: DGS via gh.com/dssg-pt/covid19pt-data', horizontalalignment='right', verticalalignment='center', color='#BBBBBB')

#####


def load_data():
    data = pd.read_csv(DATA_FILE)

    data[COL_DATE] = pd.to_datetime(
        data[COL_DATE],
        format='%d-%m-%Y %H:%M')

    return data


def extract_confirmed(data):
    return data[[COL_DATE] + [COL_TOTAL] + ['confirmados_' + k for k in REGION_COLUMNS.keys()]].copy()


def new(data, columns):
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
