import csv
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler


class Plotter:
    def __init__(self,
                 filename: str,
                 min_pressure: Optional[float] = None,
                 max_pressure: Optional[float] = None,
                 min_usage: Optional[float] = None,
                 max_leak_rate: Optional[float] = None,
                 alpha: Optional[float] = None,
                 strip_outliers: Optional[float] = None,
                 include_leak_rate: bool = False,
                 include_quadratic: bool = False,
                 ):
        self.outlier = strip_outliers if strip_outliers is not None else 0
        self.include_quadratic = include_quadratic

        df = pd.read_csv('cpap.csv').replace('--', np.nan).dropna()
        df['Usage'] = pd.to_timedelta(df['Usage'] + ':00').dt.total_seconds() / 3600

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
        numeric_fields = list(all_fields)
        df[numeric_fields] = df[numeric_fields].apply(pd.to_numeric, errors='coerce').dropna()

        if min_pressure is not None:
            df = df[df['Pressure'] >= min_pressure]
        if max_pressure is not None:
            df = df[df['Pressure'] <= max_pressure]
        if max_leak_rate is not None:
            df = df[df['AvgLR'] <= max_leak_rate]
        if min_usage is not None:
            df = df[df['Usage'] >= min_usage]

        self.data_by_pressure = defaultdict(list)
        pressure_counts = defaultdict(int)
        for _, row in df.iterrows():
            pressure = row['Pressure']
            pressure_counts[pressure] += 1
            self.data_by_pressure[pressure].append(row)

        # pressure histogram TODO refactor to use DataFrame
        self.min_pressure = min_pressure if min_pressure is not None else min(self.data_by_pressure.keys())
        self.max_pressure = max_pressure if max_pressure is not None else max(self.data_by_pressure.keys())

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
            all_correlations.append((y_field, correl))

        print()
        num_correlations = 10
        print(f'Top {num_correlations} correlations with Pressure:')
        all_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        for title, correl in all_correlations[:num_correlations]:
            print(f'- {title}: {correl:.3f}')

        # run ElasticNet analysis
        x_fields = ['Pressure', 'AvgLR'] if include_leak_rate else ['Pressure']

        X = df[x_fields]
        X = StandardScaler().fit_transform(X)

        if alpha is not None:
            print()
            print(f'Non-zero weights for {' + '.join(x_fields)} with alpha {alpha}:')
            all_zero = True
            for key in y_fields.keys():
                y = df[[key]]
                model = ElasticNet(alpha=alpha, l1_ratio=1, fit_intercept=True)
                model.fit(X, y)
                if sum(model.coef_):
                    all_zero = False
                    print(f'- {key}: Intercept={model.intercept_}, Weights={model.coef_}')
            if all_zero:
                print('- None')

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


    Plotter('cpap.csv',
            min_pressure=config['min_pressure'],
            max_pressure=config['max_pressure'],
            min_usage=config['min_usage'],
            max_leak_rate=config['max_leak_rate'],
            alpha=config['alpha'],
            strip_outliers=config['strip_outliers'],
            include_leak_rate=config['include_leak_rate'],
            include_quadratic=config['include_quadratic'],
            )
