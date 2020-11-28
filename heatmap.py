import pandas as pd
from chart import new
import locale

DATA_FILE = 'vendor/dssg_data.csv'
COL_DATE = 'data_dados'
FIRST_WEEKDAY = 'MON'
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
COL_AGE = sum([[
    'confirmados_' + k + '_m',
    'confirmados_' + k + '_f',
    'obitos_' + k + '_m',
    'obitos_' + k + '_f',
    ] for k in AGE_COLUMNS.keys()], [])

locale.setlocale(locale.LC_ALL, 'pt_PT.UTF-8')

def main():
    # load dssg data
    data = pd.read_csv(DATA_FILE)
    data[COL_DATE] = pd.to_datetime(
        data[COL_DATE],
        format='%d-%m-%Y %H:%M')
    data = new(data, COL_AGE)

    # aggregate by week
    by_week = data.groupby([pd.Grouper(key=COL_DATE, freq='W-' + FIRST_WEEKDAY)]).sum()

    # extract date column
    output_list = []
    for i, row in by_week.iterrows():
        for k, v in AGE_COLUMNS.items():
            output_list.append([
                i.strftime('%d') + ' a ' + (i + pd.Timedelta(pd.offsets.Day(6))).strftime('%d %b %Y').upper(),
                v,
                row[f'new_confirmados_' + k + '_f'] + row[f'new_confirmados_' + k + '_m'],
                row[f'new_obitos_' + k + '_f'] + row[f'new_obitos_' + k + '_m']
            ])

    output = pd.DataFrame(output_list, columns=['SEMANA', 'GRUPO ETÁRIO', 'CASOS_CONFIRMADOS', 'OBITOS'])

    # output csv
    output.to_csv('output/heatmap.csv', index=False)


if __name__ == '__main__':
    main()
