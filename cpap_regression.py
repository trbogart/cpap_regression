from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from numpy.polynomial.polynomial import Polynomial
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

        df = pd.read_csv(self.config['filename'])
        df = df.replace('--', np.nan).dropna()
        df['Usage'] = pd.to_timedelta(df['Usage'] + ':00').dt.total_seconds() / 3600

        self.y_fields = [f for field_config in self.config['y_fields'] if (f := Field(field_config)).enabled]
        self.y_field_names = [field.name for field in self.y_fields]
        df[self.y_field_names] = df[self.y_field_names].apply(pd.to_numeric, errors='coerce').dropna()

        if self.config['min_pressure'] is not None:
            df = df[df['Pressure'] >= self.config['min_pressure']]
        if self.config['max_pressure'] is not None:
            df = df[df['Pressure'] <= self.config['max_pressure']]
        if self.config['max_leak_rate'] is not None:
            df = df[df['AvgLR'] <= self.config['max_leak_rate']]
        if self.config['min_usage'] is not None:
            df = df[df['Usage'] >= self.config['min_usage']]

        self.df = df
        self.min_pressure = df['Pressure'].min()
        self.max_pressure = df['Pressure'].max()

        if self.config['weighted']:
            pressure_counts = self.df['Pressure'].value_counts().items()
            pressure_count_map = {pressure: count for pressure, count in pressure_counts}
            self.weights = [1 / pressure_count_map[pressure] for pressure in self.df['Pressure']]
        else:
            self.weights = [1.0] * len(self.df)

    def run(self):
        print(f'N={len(self.df)}')
        print('Pressure Counts:')
        for pressure, count in self.df['Pressure'].value_counts().sort_index().items():
            print(f'- {pressure:.1f}: {count}')

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
        print(f'Non-zero LASSO weights with alpha {self.config['alpha']}:')
        x = StandardScaler().fit_transform(self.df[self.y_field_names])
        y = self.df['Pressure']
        model = SGDRegressor(penalty="elasticnet", alpha=self.config['alpha'],
                             l1_ratio=1, fit_intercept=True,
                             random_state=self.config['seed'])
        model.fit(x, y, sample_weight=self.weights)
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
