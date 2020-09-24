import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import FR, MO, SA, SU, TH, TU, WE
from pylab import rcParams

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
    data = load_data()
    data_new = new(data, COL_REGION_CONFIRMED + COL_REGION_DEATHS + ['obitos'])

    plot_confirmed(data_new)
    plt.savefig('novos_casos.png')

    plot_deaths(data_new)
    plt.savefig('novos_obitos.png')

    #plot_combined(data_new)
    #plt.savefig('combinado.png')


def plot_confirmed(data):
    x = data[COL_DATE]

    fig, ax = plot_init()

    for col, label in REGION_COLUMNS.items():
        plt.plot(
            x,
            data['confirmados_' + col]
                .div(REGION_POP[col])
                .mul(100000)
                .rolling(7)
                .mean(),
            label=label)

    plt.plot(
        x,
        data[COL_TOTAL]
            .div(TOTAL_POP)
            .mul(100000)
            .rolling(7)
            .mean(),
        label='País',
        color='#000000')

    plt.legend(loc='upper left')

    title = r'$\bf{' + 'COVID19\\ Portugal' + '}$ | Novos casos / 100.000 habitantes | Média móvel de 7 dias | '
    title += data[COL_DATE].iloc[-1].strftime('%Y-%m-%d')
    plt.title(title, loc='left')

    plot_footer()


def plot_deaths(data):
    x = data[COL_DATE]

    fig, ax = plot_init()

    for col, label in REGION_COLUMNS.items():
        plt.plot(
            x,
            data['obitos_' + col]
                .div(REGION_POP[col])
                .mul(100000)
                .rolling(7)
                .mean(),
            label=label)

    plt.plot(
        x,
        data['obitos']
            .div(TOTAL_POP)
            .mul(100000)
            .rolling(7)
            .mean(),
        label='País',
        color='#000000')

    plt.legend(loc='upper left')

    title = r'$\bf{' + 'COVID19\\ Portugal' + '}$ | Novos óbitos / 100.000 habitantes | Média móvel de 7 dias | '
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
    rcParams["font.family"] = "Helvetica"
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
    ax.grid(axis='both', color='#EEEEEE')
    ax.set_xlabel('semana (segunda-feira)')
    return fig, ax


def plot_footer():
    plt.figtext(0.985, 0.915, 'Fonte: DGS via gh.com/dssg-pt/covid19pt-data', horizontalalignment='right', verticalalignment='center', color='#BBBBBB')

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

        data[col] = new

    return data


def region_new(data, prefix):
    return new(data, [prefix + k for k in REGION_COLUMNS.keys()])

if __name__ == '__main__':
    main()
