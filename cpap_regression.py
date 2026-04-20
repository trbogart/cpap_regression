import csv

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, filename: str):
        with open(filename, mode='r') as file:
            min_pressure = 7.0
            max_pressure = 8.0
            pressure_field = 'Pressure'
            y_fields = {
                'Comb FL': 'Combined FL (WAT/NED)',
                'FLS': 'Flow Limitation Score',
                'CAI': 'CAI',
                'RDI': 'RDI',
                'NED RDI': 'NED RDI',
                'NED RERA': 'NED RERA',
                'Regul': 'Regularity',
                'Period': 'Periodicity',
                'GI': 'Glasgow Index',
            }

            # initialize empty data
            self.cols = dict[str, list[float]]()
            self.cols[pressure_field] = []
            for y_field in y_fields.keys():
                self.cols[y_field] = []

            # populate data from CSV
            for row in csv.DictReader(file):
                if row['CAI']:
                    pressure = float(row[pressure_field])
                    if min_pressure <= pressure <= max_pressure:
                        self.cols[pressure_field].append(pressure)

                        for field in y_fields.keys():
                            self.cols[field].append(float(row[field]))

            # pressure histogram
            pressure_counts = {}
            for pressure in range(int(min_pressure * 10), 1 + int(max_pressure * 10), 2):
                pressure_counts[pressure / 10] = 0
            for pressure in self.cols[pressure_field]:
                pressure_counts[pressure] = pressure_counts.get(pressure, 0) + 1
            print('Pressure Counts:')
            for pressure in sorted(pressure_counts.keys()):
                print(f'{pressure:.1f}: {pressure_counts[pressure]}')

            for y_field, title in y_fields.items():
                self.plot(pressure_field, y_field, title)

    @staticmethod
    def rename_file(filename):
        return filename.lower().replace(' ', '_')

    def plot(self, x_field, y_field, title):
        if title is None:
            title = f'{y_field} vs {x_field}'

        x = self.cols[x_field]
        y = self.cols[y_field]

        # linear regression
        coef1 = np.polyfit(x, y, 1)
        model1 = np.poly1d(coef1)

        # quadratic regression
        coef2 = np.polyfit(x, y, 2)
        a, b, c = coef2
        model2 = np.poly1d(coef2)

        # plot minima or maxima of quadratic regression, if in domain
        x_extrema = -b / (2 * a)
        # noinspection PyTypeChecker
        if min(x) <= x_extrema <= max(x):
            plt.axvline(x_extrema, color='red', linestyle='--', linewidth=1)

        # plot regressions
        polyline = np.linspace(min(x), max(x), 20)
        plt.plot(polyline, model2(polyline), color='red')
        plt.plot(polyline, model1(polyline), color='blue')

        # finish plot
        plt.scatter(x, y)
        plt.xlabel(x_field)
        plt.ylabel(y_field)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(Plotter.rename_file(f'{y_field}_{x_field}.png'), bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    Plotter('cpap.csv')
