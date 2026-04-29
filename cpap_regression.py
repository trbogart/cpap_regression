import csv

import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self, filename: str, strip_outliers: float = 0, include_quadratic: bool = False):
        self.outlier = strip_outliers
        self.include_quadratic = include_quadratic
        with open(filename, mode='r') as file:
            # manually set min/max pressure
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
                # 'RHR': 'Resting Heart Rate (Oura)',
                # 'HRV': 'Heart Rate Variability (Oura)',
                # 'SpO2': 'SpO2 (Oura)',
            }

            self.data_by_pressure = {}

            # populate data from CSV
            for row in csv.DictReader(file):
                # include rows populated with data (does not support partial data) with pressure in range
                if row['CAI']:
                    pressure = float(row[pressure_field])

                    data_for_pressure = self.data_by_pressure.get(pressure, [])
                    if not data_for_pressure:
                        self.data_by_pressure[pressure] = data_for_pressure

                    data = {}
                    data_for_pressure.append(data)

                    for field in y_fields.keys():
                        data[field] = self.parse_value(row[field])

            # pressure histogram
            pressure_counts = {}
            min_pressure = min(self.data_by_pressure.keys())
            max_pressure = max(self.data_by_pressure.keys())
            for pressure in range(int(min_pressure * 10), 1 + int(max_pressure * 10), 2):
                # populate missing counts
                pressure_counts[pressure / 10] = 0
            total_count = 0
            for pressure, rows in self.data_by_pressure.items():
                pressure_counts[pressure] = pressure_counts.get(pressure, 0) + len(rows)
                total_count += len(rows)
            print(f'N={total_count}')
            print('Pressure Counts:')
            for pressure in sorted(pressure_counts.keys()):
                stripped_suffix = ''
                if strip_outliers > 0:
                    stripped = int(strip_outliers * pressure_counts[pressure])
                    if stripped > 0:
                        stripped_suffix = f' (-{stripped * 2} outliers)'
                print(f'{pressure:.1f}: {pressure_counts[pressure]}{stripped_suffix}')
            avg_pressure = np.average(list(pressure_counts.keys()), weights=list(pressure_counts.values()))
            print(f'Average Pressure: {avg_pressure :.3f}')

            self.min_pressure = min_pressure
            self.max_pressure = max_pressure
            self.polyline = np.linspace(self.min_pressure, self.max_pressure, total_count)

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

        x = []
        y = []

        for pressure, rows in self.data_by_pressure.items():
            values = []
            for row in rows:
                if y_field in row:
                    values.append(row[y_field])

            strip = int(self.outlier * len(values))
            if strip >= 1:
                values.sort()
                values = values[strip:-strip]

            for value in values:
                x.append(pressure)
                y.append(value)

        # linear regression
        coef1 = np.polyfit(x, y, 1)
        model1 = np.poly1d(coef1)

        # quadratic regression
        if self.include_quadratic:
            coef2 = np.polyfit(x, y, 2)
            a, b, c = coef2
            model2 = np.poly1d(coef2)

            # plot minima or maxima of quadratic regression, if in domain
            x_extrema = -b / (2 * a)
            if self.min_pressure <= x_extrema <= self.max_pressure:
                plt.axvline(x_extrema, color='red', linestyle='--', linewidth=1)

            plt.plot(self.polyline, model2(self.polyline), color='red')

        # linear regression
        plt.plot(self.polyline, model1(self.polyline), color='blue')

        correl = np.corrcoef(x, y)[0, 1]

        # finish plot
        plt.scatter(x, y)
        plt.xlabel(f'Pressure (r={correl:.2f})')
        plt.ylabel(y_field)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(self.field_to_filename(f'{y_field}.png'), bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    Plotter('cpap.csv')
