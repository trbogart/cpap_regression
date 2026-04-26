import csv

import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self, filename: str):
        with open(filename, mode='r') as file:
            # manually set min/max pressure
            min_pressure = 7.0
            max_pressure = 8.0
            pressure_field = 'Pressure'
            y_fields = {
                'CAI': 'CAI (AS11)',
                'RDI': 'RDI (AS11)',
                # 'Obs': 'Obstructive Event Index (AS11)',
                'Comb FL': 'Combined FL (WAT/NED)',
                'FLS': 'Flow Limitation Score (WAT)',
                # 'IFL': 'IFL Symptom Risk %',
                'NED Mean': 'NED Mean (NED)',
                'NED RERA': 'RERA/hr (NED)',
                'NED RDI': 'Est. RDI (NED)',
                'GI': 'Glasgow Index',
                'GI TH': 'Glasgow Index: Top-Heavy',
                'GI VA': 'Glasgow Index: Variable Amplitude',
                'BOI': 'Brief Obstruction Index',
                'Regul': 'Regularity (WAT)',
                'Period': 'Periodicity (WAT)',
                # 'Usage': 'Usage (AS11)',
                'RHR': 'Resting Heart Rate (Oura)',
                'HRV': 'Heart Rate Variability (Oura)',
                'SpO2': 'SpO2 (Oura)',
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
                            self.data[field].append(self.parse_value(row[field]))

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
    def parse_value(value: str) -> float:
        if ':' in value:
            # handle usage
            tokens = value.split(':')
            return float(tokens[0]) + float(tokens[1])/60
        return float(value)

    @staticmethod
    def field_to_filename(field: str):
        return field.lower().replace(' ', '_')

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
        plt.savefig(self.field_to_filename(f'{y_field}.png'), bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    Plotter('cpap.csv')
