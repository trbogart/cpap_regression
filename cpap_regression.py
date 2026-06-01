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
    # noinspection PyArgumentList
    def __init__(self, config_filename: str):
        with open('config.yaml', 'r') as file:
            # Use safe_load to avoid executing arbitrary code from the file
            self.config = yaml.safe_load(file)

        self.y_fields = [f for field_config in self.config['y_fields'] if (f := Field(field_config)).enabled]
        self.y_field_names = [field.name for field in self.y_fields]

        df = pd.read_csv(self.config['filename'])

        # pressure field can be empty or contain an exclusion note
        df['Pressure'] = pd.to_numeric(df['Pressure'], errors='coerce')
        
        # weight field is 1 by default (mostly intended for manual exclusion)
        df['Weight'] = df['Weight'].fillna(1)

        df = df.dropna()
        df = df.sort_values(by='Date')

        # convert time fields from H:MM to float hours
        df['Usage'] = pd.to_timedelta(df['Usage'] + ':00').dt.total_seconds() / 3600
        df['Sleep'] = pd.to_timedelta(df['Sleep'] + ':00').dt.total_seconds() / 3600

        # calculated fields
        df['Efficiency'] = df['Sleep'] / df['Usage']
        df['RDI'] = df['AHI'] + df['RERA']

        # filter by config or by zero weight
        df = self._filter(df)

        self.pressure = df['Pressure']
        self.min_pressure = self.pressure.min()
        self.max_pressure = self.pressure.max()

        # adjust weights based on config
        if self.config['weight_frequency']:
            pressure_counts = self.pressure.value_counts().items()
            pressure_count_map = {pressure: count for pressure, count in pressure_counts}
            df['Weight'] /= [pressure_count_map[pressure] for pressure in self.pressure]

        if self.config['weight_usage']:
            df['Weight'] *= df['Usage']

        self.df = df

    def _filter(self, df: DataFrame) -> DataFrame:
        dates = set(df['Date'])
        df, dates = self._filter_column(df, dates, 'Weight')  # filter zero weights
        df, dates = self._filter_column(df, dates, 'Pressure', 'min_pressure')
        df, dates = self._filter_column(df, dates, 'Pressure', 'max_pressure')
        df, dates = self._filter_column(df, dates, 'AvgLR', 'max_leak_rate')
        df, dates = self._filter_column(df, dates, 'Usage', 'min_usage')
        df, dates = self._filter_column(df, dates, 'Sleep', 'min_sleep')
        df, dates = self._filter_column(df, dates, 'Efficiency', 'min_sleep_efficiency')
        return df

    def _filter_column(self, df: DataFrame, dates: set, field: str, config_key: str | None = None) -> tuple[DataFrame, set]:
        if config_key is not None:
            threshold = self.config[config_key]
            if threshold is None:
                return df, dates
            if config_key.startswith('min_'):
                filtered_df = df[df[field] >= threshold]
            elif config_key.startswith('max_'):
                filtered_df = df[df[field] <= threshold]
            else:
                raise ValueError(f'Invalid config name: {config_key}')
        else:
            threshold = 0.0
            filtered_df = df[df[field] > threshold]

        new_dates = set(filtered_df['Date'])
        removed_dates = dates - new_dates
        removed_rows = df[df['Date'].isin(removed_dates)]

        num_removed = len(dates) - len(new_dates)
        if config_key is not None:
            print(f'Dropped {num_removed} rows for '
                  f'{config_key.replace('_', ' ')}: {threshold}:')
        elif num_removed > 0:
            # for weight
            print(f'Dropped {num_removed} rows with zero {field}:')
        if num_removed > 0:
            for _, row in removed_rows.iterrows():
                line = f'- {row['Date']}: Pressure={row['Pressure']:.1f}'
                if field not in {'Pressure', 'Weight'}:
                    line += f', {field}={row[field]:.2f}'
                print(line)

        return filtered_df, new_dates

    def run(self):
        print()

        print(f'N={len(self.df)}, {self._weighted_by()}')
        print('Pressure Counts:')
        for pressure in sorted(self.pressure.unique()):
            data_for_pressure = self.df[self.pressure == pressure]
            dates = data_for_pressure['Date']
            total_usage = data_for_pressure['Usage'].sum()
            total_weight = data_for_pressure['Weight'].sum()
            line = f'- {pressure:.1f} ({len(dates)}, {total_usage:.1f} hrs'
            if total_weight != len(dates):
                line +=  f', {total_weight:.2f} total weight)'
            line += f'): {', '.join(dates)}'
            print(line)

        avg_pressure = self.pressure.mean()
        print(f'Average Pressure: {avg_pressure :.3f}')

        all_correlations = []

        for field in self.y_fields:
            correl = self._calculate_field(field)
            all_correlations.append((field.name, correl))

        if self.config['num_correlations'] is not None:
            all_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

            print()
            if self.config['num_correlations'] == 0:
                print(f'Correlations with Pressure:')
            else:
                print(f'Top {self.config['num_correlations']} correlations with Pressure:')
                all_correlations = all_correlations[:self.config['num_correlations']]

            for field, correl in all_correlations:
                print(f'- {field}: {correl:.3f}')

        if self.config['alpha'] is not None:
            self._elastic_net()

    def _weighted_by(self) -> str:
        if self.config['weight_frequency']:
            if self.config['weight_usage']:
                return 'weighted by inverse frequency and usage'
            return 'weighted by inverse frequency'
        if self.config['weight_usage']:
            return 'weighted by usage'
        return 'not weighted'

    def _calculate_field(self, field: Field) -> float:
        x = self.pressure
        y = self.df[field.name]
        correl = self._weighted_correlation(x, y, self.df['Weight'])
        if field.plot and self.config['plot']:
            polyline = np.linspace(self.min_pressure, self.max_pressure, 100)

            # linear regression
            poly1 = Polynomial.fit(x, y, 1, w=self.df['Weight'])
            plt.plot(polyline, poly1(polyline), color='blue')

            # quadratic regression
            if self.config['plot_quadratic']:
                poly2 = Polynomial.fit(x, y, 2, w=self.df['Weight'])
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
                plt.savefig(self._field_filename(f'{field.name}.png'), bbox_inches='tight')
            plt.show()

        return correl

    def _elastic_net(self):
        # run ElasticNet analysis
        print()
        print(f'Non-zero ElasticNet weights with alpha {self.config['alpha']} '
              f'and l1_ratio = {self.config['l1_ratio']}:')
        X = StandardScaler().fit_transform(self.df[self.y_field_names])
        y = self.pressure
        model = SGDRegressor(penalty="elasticnet", alpha=self.config['alpha'],
                             l1_ratio=self.config['l1_ratio'], fit_intercept=True,
                             random_state=self.config['seed'])
        model.fit(X, y, sample_weight=self.df['Weight'])
        field_weights = [(self.y_field_names[i], coef) for i, coef in enumerate(model.coef_) if coef > 0]
        if field_weights:
            field_weights.sort(key=lambda x: abs(x[1]), reverse=True)
            for field, weight in field_weights:
                print(f'- {field}: {weight:.3f}')
        else:
            print('- None')

    @staticmethod
    def _field_filename(field: str):
        return field.lower().replace(' ', '_')

    @staticmethod
    def _weighted_correlation(x, y, weights):
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
