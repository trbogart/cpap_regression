import csv

import matplotlib.pyplot as plt
import numpy as np


def rename_file(filename):
    return filename.lower().replace(' ', '_')


def plot(data, x_field, y_field, title):
    if title is None:
        title = f'{y_field} vs {x_field}'

    x_data = []
    y_data = []
    for row in data:
        if x_field in row and y_field in row:
            x_data.append(float(row[x_field]))
            y_data.append(float(row[y_field]))

    x = np.array(x_data)
    y = np.array(y_data)

    # linear regression
    coef1 = np.polyfit(x, y, 1)
    model1 = np.poly1d(coef1)

    # quadratic regression
    coef2 = np.polyfit(x, y, 2)
    a, b, c = coef2
    model2 = np.poly1d(coef2)

    # plot minima or maxima, if in domain
    x_min = -b / (2 * a)
    if min(x) <= x_min <= max(x):
        plt.axvline(x_min, color='red', linestyle='--', linewidth=1)

    # Plot
    polyline = np.linspace(min(x), max(x), 20)
    plt.scatter(x, y)
    plt.plot(polyline, model2(polyline), color='red')
    plt.plot(polyline, model1(polyline), color='blue')
    plt.xlabel(x_field)
    plt.ylabel(y_field)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(rename_file(f'{y_field}_{x_field}.png'), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    with open('cpap.csv', mode='r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]

        min_pressure = 7.0
        max_pressure = 8.0

        data = [row for row in data if min_pressure <= float(row['Pressure']) <= max_pressure]

        pressure_counts = {}
        for pressure in range(int(min_pressure * 10), 1 + int(max_pressure * 10), 2):
            pressure_counts[pressure / 10] = 0
        for row in data:
            if 'Pressure' in row:
                pressure = float(row['Pressure'])
                pressure_counts[pressure] = pressure_counts.get(pressure, 0) + 1
        print('Pressure Counts:')
        for pressure in sorted(pressure_counts.keys()):
            print(f'{pressure:.1f}: {pressure_counts[pressure]}')

        y_fields = {
            'Comb FL': 'Combined FL (WAT/NED)',
            'FLS': 'Flow Limitation Score',
            'CAI': 'CAI',
            'NED RDI': 'NED RDI',
            'Regul': 'Regularity',
            'Period': 'Periodicity',
            'GI': 'Glasgow Index',
        }

        for y_field, title in y_fields.items():
            plot(data, 'Pressure', y_field, title)
