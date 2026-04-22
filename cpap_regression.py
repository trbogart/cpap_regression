import csv

import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self, filename: str):
        with open(filename, mode='r') as file:
            # manually set min/max pressure
            min_pressure = 6.8
            max_pressure = 8.0
            pressure_field = 'Pressure'
            y_fields = {
                'CAI': 'CAI (AS11)',
                'Comb FL': 'Combined FL (WAT/NED)',
                'FLS': 'Flow Limitation Score (WAT)',
                # 'IFL': 'IFL Symptom Risk %',
                # 'Regul': 'Regularity (WAT)',
                # 'Period': 'Periodicity (WAT)',
                # 'NED Mean': 'NED Mean (NED)',
                # 'NED RERA': 'RERA Index (NED)',
                'GI': 'Glasgow Index: Overall',
                # 'GI TH': 'Glasgow Index: Top-Heavy',
                # 'GI VA': 'Glasgow Index: Variable Amplitude',
            }

            # initialize empty data (column-based)
            self.data = {field: list[float]() for field in y_fields.keys()}
            self.pressures = list[float]()

            # populate data from CSV
            for row in csv.DictReader(file):
                # include rows populated with data (does not support partial data) with pressure in range
                if row['CAI']:
                    pressure = float(row[pressure_field])
                    if min_pressure <= pressure <= max_pressure:
                        self.pressures.append(pressure)

                        for field in y_fields.keys():
                            self.data[field].append(float(row[field]))

            # pressure histogram
            pressure_counts = dict[float, int]()
            for pressure in range(int(min_pressure * 10), 1 + int(max_pressure * 10), 2):
                # populate missing counts
                pressure_counts[pressure / 10] = 0
            for pressure in self.pressures:
                pressure_counts[pressure] = pressure_counts.get(pressure, 0) + 1
            print(f'N={len(self.pressures)}')
            print('Pressure Counts:')
            for pressure in sorted(pressure_counts.keys()):
                print(f'{pressure:.1f}: {pressure_counts[pressure]}')
            print(f'Average Pressure: {np.average(self.pressures):.3f}')

            self.min_pressure = min_pressure
            self.max_pressure = max_pressure
            self.polyline = np.linspace(self.min_pressure, self.max_pressure, len(self.pressures))

            for y_field, title in y_fields.items():
                self.plot(y_field, title)

    @staticmethod
    def rename_file(filename: str):
        return filename.lower().replace(' ', '_')

    def plot(self, y_field: str, title: str):
        title = title if title else y_field

        y = self.data[y_field]

        # linear regression
        coef1 = np.polyfit(self.pressures, y, 1)
        model1 = np.poly1d(coef1)

        # quadratic regression
        coef2 = np.polyfit(self.pressures, y, 2)
        a, b, c = coef2
        model2 = np.poly1d(coef2)

        # plot minima or maxima of quadratic regression, if in domain
        x_extrema = -b / (2 * a)
        if self.min_pressure <= x_extrema <= self.max_pressure:
            plt.axvline(x_extrema, color='red', linestyle='--', linewidth=1)

        # plot regressions
        plt.plot(self.polyline, model2(self.polyline), color='red')
        plt.plot(self.polyline, model1(self.polyline), color='blue')

        # finish plot
        plt.scatter(self.pressures, y)
        plt.xlabel('Pressure')
        plt.ylabel(y_field)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(Plotter.rename_file(f'{y_field}.png'), bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    Plotter('cpap.csv')
