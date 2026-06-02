from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from numpy.polynomial.polynomial import Polynomial
from sklearn.linear_model import SGDRegressor, BayesianRidge
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
        with open(config_filename, 'r') as file:
            # Use safe_load to avoid executing arbitrary code from the file
            self.config = yaml.safe_load(file)

        self.y_fields = [f for field_config in self.config['y_fields'] if (f := Field(field_config)).enabled]

        self.df = pd.read_csv(self.config['filename'])

        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
        self.df = self.df.sort_values(by='Date')
        if self.config['max_days']:
            self.df = self.df.tail(self.config['max_days'])

        # pressure field can be empty or contain an exclusion note
        self.df['Pressure'] = pd.to_numeric(self.df['Pressure'], errors='coerce')

        # weight field is 1 by default (mostly intended for manual exclusion)
        self.df['Weight'] = self.df['Weight'].fillna(1)

        self.df = self.df.dropna()

        # convert time fields from H:MM to float hours
        self.df['Usage'] = pd.to_timedelta(self.df['Usage'] + ':00').dt.total_seconds() / 3600
        self.df['Sleep'] = pd.to_timedelta(self.df['Sleep'] + ':00').dt.total_seconds() / 3600
        self.df['REM'] = pd.to_timedelta(self.df['REM'] + ':00').dt.total_seconds() / 3600
        self.df['Deep'] = pd.to_timedelta(self.df['Deep'] + ':00').dt.total_seconds() / 3600

        # calculated fields
        self.df['Efficiency'] = self.df['Sleep'] / self.df['Usage']
        self.df['RDI'] = self.df['AHI'] + self.df['RERA']

        self.log_file = open(self._log_filename(), 'w') if self.config['save_logs'] else None

        # filter by config or by zero weight (possible with manual weighting)
        self._filter()
        self.pressure = self.df['Pressure']
        self.min_pressure = self.pressure.min()
        self.max_pressure = self.pressure.max()
        self.multi_x_field_names = self.config['multi_x_fields']
        self.multi_x = StandardScaler().fit_transform(self.df[self.multi_x_field_names])

        # adjust weights based on config
        if self.config['weight_frequency']:
            pressure_counts = self.pressure.value_counts().items()
            pressure_count_map = {pressure: count for pressure, count in pressure_counts}
            self.df['Weight'] /= [pressure_count_map[pressure] for pressure in self.pressure]

        if self.config['weight_usage']:
            self.df['Weight'] *= self.df['Usage']
        self.multi_y_fields = [field for field in self.y_fields if field.name not in self.multi_x_field_names]

    def log(self, s: str):
        print(s)
        if self.log_file:
            print(s, file=self.log_file)

    def _filter(self):
        # noinspection PyStringConversionWithoutDunderMethod
        self.log(f'Unfiltered N={len(self.df)} between {self.df['Date'].min()} and {self.df['Date'].max()}')
        dates = set(self.df['Date'])
        dates = self._filter_column(dates, 'Weight')  # filter zero weights
        dates = self._filter_column(dates, 'Pressure', 'min_pressure')
        dates = self._filter_column(dates, 'Pressure', 'max_pressure')
        dates = self._filter_column(dates, 'AvgLR', 'max_leak_rate')
        dates = self._filter_column(dates, 'Usage', 'min_usage')
        dates = self._filter_column(dates, 'Sleep', 'min_sleep')
        self._filter_column(dates, 'Efficiency', 'min_sleep_efficiency')
        print()

    def _filter_column(self, dates: set, field: str, config_key: str | None = None) -> set:
        if config_key is not None:
            threshold = self.config[config_key]
            if threshold is None:
                return dates
            if config_key.startswith('min_'):
                filtered_df = self.df[self.df[field] >= threshold]
            elif config_key.startswith('max_'):
                filtered_df = self.df[self.df[field] <= threshold]
            else:
                raise ValueError(f'Invalid config name: {config_key}')
        else:
            threshold = 0.0
            filtered_df = self.df[self.df[field] > threshold]

        new_dates = set(filtered_df['Date'])
        removed_dates = dates - new_dates
        removed_rows = self.df[self.df['Date'].isin(removed_dates)]

        num_removed = len(dates) - len(new_dates)
        if config_key is not None:
            self.log(f'Dropped {num_removed} rows for '
                  f'{config_key.replace('_', ' ')}: {threshold}:')
        elif num_removed > 0:
            # for weight
            self.log(f'Dropped {num_removed} rows with zero {field}:')
        if num_removed > 0:
            for _, row in removed_rows.iterrows():
                line = f'- {row['Date']}: Pressure={row['Pressure']:.1f}'
                if field not in {'Pressure', 'Weight'}:
                    line += f', {field}={row[field]:.2f}'
                self.log(line)

        self.df = filtered_df
        return new_dates

    def run(self):
        self.log(f'Filtered N={len(self.df)} between {self.df['Date'].min()} and {self.df['Date'].max()}, '
                 f'{self._weighted_by()}')
        self.log('Pressure Counts:')
        for pressure in sorted(self.pressure.unique()):
            data_for_pressure = self.df[self.pressure == pressure]
            dates = data_for_pressure['Date']
            total_usage = data_for_pressure['Usage'].sum()
            total_weight = data_for_pressure['Weight'].sum()
            self.log(f'- {pressure:.1f} ({len(dates)}, {total_usage:.1f} hrs, '
                     f'{total_weight:.2f} total weight): {', '.join(dates)})')

        avg_pressure = self.pressure.mean()
        self.log(f'Mean Pressure: {avg_pressure :.3f}')

        # Correlation and linear regression
        all_correlations = [(field.name, self._linear(field)) for field in self.y_fields]
        self.log('\nCorrelations with pressure:')
        self._print_field_weights(all_correlations)

        if self.config['alpha'] is not None:
            self._elastic_net()

        if self.config['bayesian']:
            self._bayesian()

    def _weighted_by(self, include_unweighted: bool = True) -> str | None:
        if self.config['weight_frequency']:
            if self.config['weight_usage']:
                return 'weighted by inverse frequency and usage'
            return 'weighted by inverse frequency'
        if self.config['weight_usage']:
            return 'weighted by usage'
        if include_unweighted:
            return 'not weighted'
        return None

    def _print_field_weights(self, fields_and_weights: list[tuple[str, float]], prefix: str = ''):
        fields_and_weights.sort(key=lambda x: abs(x[1]), reverse=True)
        if self.config['num_correlations'] > 0:
            fields_and_weights = fields_and_weights[:self.config['num_correlations']]
        for field, weight in fields_and_weights:
            self.log(f'{prefix}- {field}: {weight:.3f}')

    def _linear(self, field: Field) -> float:
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

            weighted_by = self._weighted_by(include_unweighted=False)
            if weighted_by:
                title = f'{field.title} - {weighted_by}'
            else:
                title = field.title

            plt.scatter(x, y)
            plt.xlabel(f'Pressure (r = {correl:.2f})')
            plt.ylabel(field.name)
            plt.title(title)
            plt.tight_layout()
            if self.config['save_plots']:
                plt.savefig(self._plot_filename(field), bbox_inches='tight')
            plt.show()

        return correl

    def _elastic_net(self):
        self.log(f'\nNon-zero ElasticNet weights with alpha {self.config['alpha']} '
                 f'and l1_ratio = {self.config['l1_ratio']}:')
        for field in self.multi_y_fields:
            model = SGDRegressor(penalty="elasticnet", alpha=self.config['alpha'],
                                 l1_ratio=self.config['l1_ratio'], fit_intercept=True,
                                 random_state=self.config['seed'])
            model.fit(self.multi_x, self.df[field.name], sample_weight=self.df['Weight'])
            self._print_multi_field_weights(field, model.coef_)

    def _bayesian(self):
        min_weight = self.config['min_bayesian_weight'] if self.config['min_bayesian_weight'] else 0
        self.log(f'\nBayesian weights with magnitude > {min_weight}:')
        for field in self.multi_y_fields:
            model = BayesianRidge()
            model.fit(self.multi_x, self.df[field.name], sample_weight=self.df['Weight'])
            self._print_multi_field_weights(field, model.coef_, min_weight)

    def _print_multi_field_weights(self, field: Field, weights: np.ndarray, min_weight: float = 0):
        field_weights = [(self.multi_x_field_names[i], weight) for i, weight in enumerate(weights) if
                         abs(weight) > min_weight]
        if field_weights:
            self.log(f'- {field.name}:')
            self._print_field_weights(field_weights, prefix='  ')

    def _plot_filename(self, field: Field):
        s = []
        if self.config['weight_frequency']:
            s.append('freq')
        if self.config['weight_usage']:
            s.append('usage')
        s.append(str(self.df['Date'].min()))
        s.append('to')
        s.append(str(self.df['Date'].max()))
        return f'{field.name.lower().replace(' ', '_')}_{self._base_filename()}.png'

    def _log_filename(self):
        return f'results_{self._base_filename()}.txt'

    def _base_filename(self) -> str:
        s = [str(self.df['Date'].min()), 'to', str(self.df['Date'].max())]
        if self.config['weight_frequency']:
            s.append('freq')
        if self.config['weight_usage']:
            s.append('usage')
        return '_'.join(s)

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
