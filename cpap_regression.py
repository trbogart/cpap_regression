import csv
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml


class Plotter:
    def __init__(self,
                 filename: str,
                 min_pressure_opt: Optional[float] = None,
                 max_pressure_opt: Optional[float] = None,
                 min_usage_opt: Optional[float] = None,
                 max_leak_rate_opt: Optional[float] = None,
                 strip_outliers: float = 0.0,
                 include_quadratic: bool = False,
                 ):
        self.outlier = strip_outliers
        self.include_quadratic = include_quadratic

        with open(filename, mode='r') as file:
            # manually set min/max pressure
            pressure_field = 'Pressure'
            y_fields = {
                'Comb FL': 'Combined FL (WAT/NED)',
                'FLS': 'Flow Limitation Score (WAT)',
                'GI': 'Glasgow Index',
                'NED RDI': 'Est. RDI (NED)',
                'BOI': 'Brief Obstruction Index',
            }

            other_y_fields = {
                'Usage': 'Usage (AS11)',
                'AHI': 'AHI (AS11)',
                'CAI': 'CAI (AS11)',
                'RDI': 'RDI (AS11)',
                '95%FL': '95% Flow Limitation (AS11)',
                'AvgLR': 'Average Leak Rate (AS11)',
                'IFL': 'IFL Symptom Risk % (WAT/NED/GI)',
                'NED Mean': 'NED Mean (NED)',
                'NED RERA': 'RERA/hr (NED)',
                'GI TH': 'Glasgow Index: Top-Heavy',
                'GI VA': 'Glasgow Index: Variable Amplitude',
                'Period': 'Periodicity (WAT)',
                'Regul': 'Regularity (WAT)',
                'RHR': 'Resting Heart Rate (Oura)',
                'HRV': 'Heart Rate Variability (Oura)',
                'SpO2': 'SpO2 (Oura)',
            }

            all_fields = set(y_fields.keys()).union(other_y_fields.keys())

            self.data_by_pressure = {}

            # populate data from CSV
            for row in csv.DictReader(file):
                # include rows populated with data (does not support partial data) with pressure in range
                if row['Pressure']:
                    pressure = float(row[pressure_field])

                    if min_pressure_opt is not None and pressure < min_pressure_opt:
                        print(f'Exclude {row['Date']}: Pressure={pressure}')
                        continue
                    if max_pressure_opt is not None and pressure > max_pressure_opt:
                        print(f'Exclude {row['Date']}: Pressure={pressure}')
                        continue
                    if min_usage_opt is not None and self.parse_value(row['Usage']) < min_usage_opt:
                        print(f'Exclude {row['Date']}: Usage={row['Usage']}')
                        continue
                    if max_leak_rate_opt is not None and self.parse_value(row['AvgLR']) > max_leak_rate_opt:
                        print(f'Exclude {row['Date']}: AvgLR={row['AvgLR']}')
                        continue

                    data_for_pressure = self.data_by_pressure.get(pressure, [])
                    if not data_for_pressure:
                        self.data_by_pressure[pressure] = data_for_pressure

                    data = {}
                    data_for_pressure.append(data)

                    for field in all_fields:
                        data[field] = self.parse_value(row[field])

            # pressure histogram
            pressure_counts = {}
            self.min_pressure = min_pressure_opt if min_pressure_opt is not None else min(self.data_by_pressure.keys())
            self.max_pressure = max_pressure_opt if max_pressure_opt is not None else max(self.data_by_pressure.keys())

            for pressure in range(int(self.min_pressure * 10), 1 + int(self.max_pressure * 10), 2):
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

            all_correlations = []

            for y_field, title in y_fields.items():
                correl = self.plot(y_field, title)
                all_correlations.append((title, correl))

            for y_field, title in other_y_fields.items():
                correl = self.plot(y_field, title, plot=False)
                all_correlations.append((title, correl))

            print()
            print('Correlations with pressure:')
            all_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            for title, correl in all_correlations:
                print(f'- {title}: {correl:.3f}')

    @staticmethod
    def parse_value(value: str) -> float:
        if ':' in value:
            # handle usage
            tokens = value.split(':')
            return float(tokens[0]) + float(tokens[1]) / 60
        return float(value)

    @staticmethod
    def field_to_filename(field: str):
        return field.lower().replace(' ', '_')

    def plot(self, y_field: str, title: str, plot: bool = True) -> float:
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

        if plot:
            # linear regression
            coef1 = np.polyfit(x, y, 1)
            model1 = np.poly1d(coef1)
            polyline = np.linspace(self.min_pressure, self.max_pressure, len(x))

            # quadratic regression
            if self.include_quadratic:
                coef2 = np.polyfit(x, y, 2)
                a, b, c = coef2
                model2 = np.poly1d(coef2)

                # plot minima or maxima of quadratic regression, if in domain
                x_extrema = -b / (2 * a)
                if self.min_pressure <= x_extrema <= self.max_pressure:
                    plt.axvline(x_extrema, color='red', linestyle='--', linewidth=1)

                plt.plot(polyline, model2(polyline), color='red')

            # linear regression
            plt.plot(polyline, model1(polyline), color='blue')

        correl = np.corrcoef(x, y)[0, 1]

        if plot:
            # finish plot
            plt.scatter(x, y)
            plt.xlabel(f'Pressure (r = {correl:.2f})')
            plt.ylabel(y_field)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(self.field_to_filename(f'{y_field}.png'), bbox_inches='tight')
            plt.show()

        return correl


if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        # Use safe_load to avoid executing arbitrary code from the file
        config = yaml.safe_load(file)


    def config_float(key):
        s = config.get(key)
        return float(s) if s is not None else None


    Plotter('cpap.csv',
            min_pressure_opt=config_float('min_pressure'),
            max_pressure_opt=config_float('max_pressure'),
            min_usage_opt=config_float('min_usage'),
            max_leak_rate_opt=config_float('max_leak_rate'),
            strip_outliers=0.0,
            include_quadratic=False
            )
