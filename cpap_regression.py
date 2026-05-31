from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from numpy.polynomial.polynomial import Polynomial
from pandas import DataFrame
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class Field:
    name: str
    title: str
    plot: bool

    def __init__(self, field_config: dict):
        self.name = field_config['name']
        self.title = field_config.get('title', self.name)
        self.plot = field_config['plot'] if 'plot' in field_config else False
        self.enabled = field_config['enabled'] if 'enabled' in field_config else True


class Regression:
    def __init__(self, config_filename: str):
        with open('config.yaml', 'r') as file:
            # Use safe_load to avoid executing arbitrary code from the file
            self.config = yaml.safe_load(file)

        self.y_fields = [f for field_config in self.config['y_fields'] if (f := Field(field_config)).enabled]
        self.y_field_names = [field.name for field in self.y_fields]

        df = pd.read_csv(self.config['filename'])
        df['Pressure'] = pd.to_numeric(df['Pressure'], errors='coerce')
        df = df.dropna()

        df['Usage'] = pd.to_timedelta(df['Usage'] + ':00').dt.total_seconds() / 3600
        df['Sleep'] = pd.to_timedelta(df['Sleep'] + ':00').dt.total_seconds() / 3600

        df['Efficiency'] = df['Sleep'] / df['Usage']
        df['RDI'] = df['AHI'] + df['RERA']

        self.dates = set(df['Date'])

        def filter(df: DataFrame, config: str, field: str) -> DataFrame:
            threshold = self.config[config]
            if threshold is None:
                return df
            if config.startswith('min_'):
                filtered_df = df[df[field] >= threshold]
            elif config.startswith('max_'):
                filtered_df = df[df[field] <= threshold]
            else:
                raise ValueError(f'Invalid config: {config}')

            new_dates = set(filtered_df['Date'])
            removed_dates = self.dates - new_dates
            removed_rows = df[df['Date'].isin(removed_dates)]
            removed = [f'{row['Date']} ({row[field]:.2f})' for _, row in removed_rows.iterrows()]

            print(f'Dropped {len(self.dates) - len(new_dates)} rows for '
                  f'{config.replace('_', ' ')} ({threshold}): {', '.join(removed)}')
            self.dates = new_dates
            return filtered_df

        df = filter(df, 'min_pressure', 'Pressure')
        df = filter(df, 'max_pressure', 'Pressure')

        df = filter(df, 'max_leak_rate', 'AvgLR')
        df = filter(df, 'min_usage', 'Usage')
        df = filter(df, 'min_sleep', 'Sleep')
        df = filter(df, 'min_sleep_efficiency', 'Efficiency')

        self.min_pressure = df['Pressure'].min()
        self.max_pressure = df['Pressure'].max()

        if self.config['weighted']:
            pressure_counts = df['Pressure'].value_counts().items()
            pressure_count_map = {pressure: count for pressure, count in pressure_counts}
            self.weights = [1 / pressure_count_map[pressure] for pressure in df['Pressure']]
        else:
            self.weights = [1.0] * len(df)

        self.df = df

    def run(self):
        print()
        print(f'N={len(self.df)}, weighted={self.config['weighted']}')
        print('Pressure Counts:')
        for pressure in sorted(self.df['Pressure'].unique()):
            dates = self.df[self.df['Pressure'] == pressure]['Date']
            print(f'- {pressure:.1f} ({len(dates)}): {', '.join(dates)}')

        avg_pressure = self.df['Pressure'].mean()
        print(f'Average Pressure: {avg_pressure :.3f}')

        all_correlations = []

        for field in self.y_fields:
            correl = self.calculate_field(field)
            all_correlations.append((field.name, correl))

        if self.config['num_correlations']:
            print()
            print(f'Top {self.config['num_correlations']} correlations with Pressure:')
            all_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            for field, correl in all_correlations[:self.config['num_correlations']]:
                print(f'- {field}: {correl:.3f}')

        if self.config['alpha'] is not None:
            self.elastic_net()

    def calculate_field(self, field: Field) -> float:
        x = self.df['Pressure']
        y = self.df[field.name]
        correl = self.weighted_correlation(x, y, self.weights)
        if field.plot and self.config['plot']:
            polyline = np.linspace(self.min_pressure, self.max_pressure, 100)

            # linear regression
            poly1 = Polynomial.fit(x, y, 1, w=self.weights)
            plt.plot(polyline, poly1(polyline), color='blue')

            # quadratic regression
            if self.config['plot_quadratic']:
                poly2 = Polynomial.fit(x, y, 2, w=self.weights)
                c, b, a = poly2.convert().coef

                # plot minima or maxima of quadratic regression, if in domain
                x_extrema = -b / (2 * a)
                if self.min_pressure <= x_extrema <= self.max_pressure:
                    plt.axvline(x_extrema, color='red', linestyle='--', linewidth=1)

                plt.plot(polyline, poly2(polyline), color='red')

            plt.scatter(x, y)
            plt.xlabel(f'Pressure (r = {correl:.2f})')
            plt.ylabel(field.name)
            plt.title(field.title)
            plt.tight_layout()
            if self.config['save_plots']:
                plt.savefig(self.field_to_filename(f'{field.name}.png'), bbox_inches='tight')
            plt.show()

        return correl

    def elastic_net(self):
        # run ElasticNet analysis
        print()
        print(
            f'Non-zero ElasticNet weights with alpha {self.config['alpha']} and l1_ratio = {self.config['l1_ratio']}:')
        X = StandardScaler().fit_transform(self.df[self.y_field_names])
        y = self.df['Pressure']
        model = SGDRegressor(penalty="elasticnet", alpha=self.config['alpha'],
                             l1_ratio=self.config['l1_ratio'], fit_intercept=True,
                             random_state=self.config['seed'])
        model.fit(X, y, sample_weight=self.weights)
        field_weights = [(self.y_field_names[i], coef) for i, coef in enumerate(model.coef_) if coef > 0]
        if field_weights:
            field_weights.sort(key=lambda x: abs(x[1]), reverse=True)
            for field, weight in field_weights:
                print(f'- {field}: {weight:.3f}')
        else:
            print('- None')

    @staticmethod
    def field_to_filename(field: str):
        return field.lower().replace(' ', '_')

    @staticmethod
    def weighted_correlation(x, y, weights):
        """Calculates the weighted Pearson correlation coefficient."""
        # Compute weighted means
        mean_x = np.average(x, weights=weights)
        mean_y = np.average(y, weights=weights)

        # Compute weighted covariances
        cov_xx = np.average((x - mean_x) ** 2, weights=weights)
        cov_yy = np.average((y - mean_y) ** 2, weights=weights)
        cov_xy = np.average((x - mean_x) * (y - mean_y), weights=weights)

        # Compute correlation
        return cov_xy / np.sqrt(cov_xx * cov_yy)


if __name__ == '__main__':
    Regression('config.yaml').run()
