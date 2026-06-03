import random
import sys
from dataclasses import dataclass
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from numpy.polynomial.polynomial import Polynomial
from sklearn.linear_model import ARDRegression, BayesianRidge, SGDRegressor
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Field:
    key: str
    name: str
    title: str
    plot: bool
    y_field: bool
    x_field: bool
    multi_x_field: bool

    @property
    def enabled(self) -> bool:
        return self.y_field or self.x_field or self.multi_x_field

    @property
    def multi_y_field(self) -> bool:
        return self.y_field and not self.multi_x_field

    @classmethod
    def from_config(cls, field_config: dict):
        key = field_config['key']
        name = field_config['name'] if 'name' in field_config else key
        title = field_config['title'] if 'title' in field_config else name
        plot = field_config['plot'] if 'plot' in field_config else False
        y_field = field_config['y_field'] if 'y_field' in field_config else False
        x_field = field_config['x_field'] if 'x_field' in field_config else False
        multi_x_field = field_config['multi_x_field'] if 'multi_x_field' in field_config else False
        if x_field and y_field:
            raise ValueError(f'Field {key} enabled for both x and y field')
        return cls(key, name, title, plot, y_field, x_field, multi_x_field)


class Regression:
    # noinspection PyArgumentList
    def __init__(self, config_filename: str):
        with open(config_filename, 'r') as file:
            # Use safe_load to avoid executing arbitrary code from the file
            self.config = yaml.safe_load(file)

        self.all_fields = [Field.from_config(field_config) for field_config in self.config['fields']]
        self.enabled_fields = [field for field in self.all_fields if field.enabled]
        assert len(self.enabled_fields) == len(set([field.key for field in self.enabled_fields]))

        self.y_fields = [field for field in self.enabled_fields if field.y_field]
        self.x_fields = [field for field in self.enabled_fields if field.x_field]
        self.multi_x_fields = [field for field in self.enabled_fields if field.multi_x_field]
        self.multi_y_fields = [field for field in self.enabled_fields if field.multi_y_field]

        self.df = pd.read_csv(self.config['data_file'])

        self.df['DateTime'] = pd.to_datetime(self.df['Date'], format='%m/%d/%Y')
        self.df['Timestamp'] = self.df['DateTime'].astype('int64') / 1e9
        self.df['Date'] = self.df['DateTime'].dt.strftime('%Y-%m-%d')
        self.df.sort_values(by='DateTime', inplace=True)

        filter_config = self.config['filter']
        count = len(self.df)
        if filter_config['min_date']:
            min_date = pd.to_datetime(filter_config['min_date'], format='%Y-%m-%d')
            self.df.drop(self.df[self.df['DateTime'] < min_date].index, inplace=True)
        if filter_config['max_date']:
            max_date = pd.to_datetime(filter_config['max_date'], format='%Y-%m-%d')
            self.df.drop(self.df[self.df['DateTime'] > max_date].index, inplace=True)
        if filter_config['max_days']:
            # trim data for max days (before drop NA)
            self.df = self.df.tail(filter_config['max_days'])

        # Dates have to be known before opening log file (ignoring value filtering)
        self.log_file = open(self._log_filename(), 'w') if self.config['save_logs'] else None
        new_count = len(self.df)
        self._log(f'{count} rows')
        if new_count < count:
            self._log(f'Dropped {count - len(self.df)} rows based on Date')
            count = new_count
        self.min_date_time = self.df['DateTime'].iloc[0]
        self.max_date_time = self.df['DateTime'].iloc[-1]

        # Pressure field can be empty or contain an exclusion note
        self.df['Pressure'] = pd.to_numeric(self.df['Pressure'], errors='coerce')

        # Weight field is 1 by default (mostly intended for manual exclusion, but non-zero weights also work)
        self.df['Weight'] = self.df['Weight'].fillna(1)

        # drop invalid data (including 0 weight, which is possible with manual weighting)
        self.df.drop(self.df[self.df['Weight'] == 0].index, inplace=True)
        self.df.dropna(inplace=True)
        new_count = len(self.df)
        if new_count < count:
            self._log(f'Dropped {count - len(self.df)} rows with invalid data')
            count = new_count

        if count == 0:
            print('No data')
            sys.exit(1)

        # get last pressure before config filtering so next pressure logic will work correctly
        self.last_pressure = self.df['Pressure'].iloc[-1]

        # convert time fields from H:MM to float hours
        self.df['Usage'] = pd.to_timedelta(self.df['Usage'] + ':00').dt.total_seconds() / 3600
        self.df['Sleep'] = pd.to_timedelta(self.df['Sleep'] + ':00').dt.total_seconds() / 3600
        self.df['REM'] = pd.to_timedelta(self.df['REM'] + ':00').dt.total_seconds() / 3600
        self.df['Deep'] = pd.to_timedelta(self.df['Deep'] + ':00').dt.total_seconds() / 3600

        # calculated fields
        self.df['Efficiency'] = self.df['Sleep'] / self.df['Usage']
        self.df['RDI'] = self.df['AHI'] + self.df['RERA']

        # filter by config
        dates = set(self.df['Date'])
        dates = self._filter_config(dates, 'Pressure', 'min_pressure')
        dates = self._filter_config(dates, 'Pressure', 'max_pressure')
        dates = self._filter_config(dates, 'AvgLR', 'max_leak_rate')
        dates = self._filter_config(dates, 'Usage', 'min_usage')
        dates = self._filter_config(dates, 'Sleep', 'min_sleep')
        dates = self._filter_config(dates, 'Efficiency', 'min_sleep_efficiency')
        if not self.config['filter']['verbose'] and len(dates) < count:
            self._log(f'Dropped {count - len(dates)} rows with configured filters')

        self.pressure = self.df['Pressure']
        # noinspection PyTypeChecker
        self.min_pressure: float = filter_config['min_pressure'] if filter_config[
            'min_pressure'] else self.pressure.min()
        # noinspection PyTypeChecker
        self.max_pressure: float = filter_config['max_pressure'] if filter_config[
            'max_pressure'] else self.pressure.max()
        self.valid_pressures = [p / 5 for p in range(int(self.min_pressure * 5), int(self.max_pressure * 5) + 1)]

        self.multi_x_scaled = StandardScaler().fit_transform(self.df[[field.key for field in self.multi_x_fields]])
        # adjust weights based on config
        if self.config['weighted_by']['frequency']:
            pressure_counts = self.pressure.value_counts()
            self.df['Weight'] /= [pressure_counts[pressure] for pressure in self.pressure]

        if self.config['weighted_by']['usage']:
            self.df['Weight'] *= self.df['Usage']

    def _log(self, s: str):
        print(s)
        if self.log_file:
            print(s, file=self.log_file)

    def _dates_string(self):
        days = (self.max_date_time - self.min_date_time).days + 1
        return f'between {self.min_date_time.strftime('%Y-%m-%d')} and {self.max_date_time.strftime('%Y-%m-%d')} ({days} days)'

    def _filter_config(self, dates: set, field: str, config_key: str) -> set:
        threshold = self.config['filter'][config_key]
        if threshold is None:
            return dates
        if config_key.startswith('min_'):
            filtered_df = self.df[self.df[field] >= threshold]
        elif config_key.startswith('max_'):
            filtered_df = self.df[self.df[field] <= threshold]
        else:
            raise ValueError(f'Invalid config name: {config_key}')

        new_dates = set(filtered_df['Date'])
        removed_dates = dates - new_dates
        removed_rows = self.df[self.df['Date'].isin(removed_dates)]

        num_removed = len(dates) - len(new_dates)
        if self.config['filter']['verbose']:
            if config_key is not None:
                self._log(f'Dropped {num_removed} rows for '
                          f'{config_key.replace('_', ' ')}: {threshold}')
            elif num_removed > 0:
                # for weight
                self._log(f'Dropped {num_removed} rows with zero {field}')
            if num_removed > 0:
                for _, row in removed_rows.iterrows():
                    line = f'- {row['Date']}: Pressure={row['Pressure']:.1f}'
                    if field != 'Pressure':
                        line += f', {field}={row[field]:.2f}'
                    self._log(line)

        self.df = filtered_df
        return new_dates

    def run(self):
        # noinspection PyStringConversionWithoutDunderMethod
        days = (self.max_date_time - self.min_date_time).days + 1
        date_string = f'{self.min_date_time.strftime('%Y-%m-%d')} and {self.max_date_time.strftime('%Y-%m-%d')} '

        self._log(f'\nN={len(self.df)} between {date_string} ({days} days) - {self._weighted_by()}')
        self._log('Pressure Counts:')
        for pressure in self.valid_pressures:
            data_for_pressure = self.df[self.pressure == pressure]
            dates = data_for_pressure['Date']
            total_usage = data_for_pressure['Usage'].sum()
            total_weight = data_for_pressure['Weight'].sum()
            self._log(f'- {pressure:.1f} ({len(dates)} count, {total_usage:.1f} hrs, '
                      f'{total_weight:.2f} total weight): {', '.join(dates)}')

        if self.config['next_pressure']['enabled']:
            self._next_pressure()

        if len(self.df) < 2:
            self._log(f'Minimum N=2')
            sys.exit(0)

        if self.config['all_correlations']['enabled']:
            self._all_correlations()

        # Correlation and linear regression
        if self.config['linear']['enabled']:
            self._linear()

        if self.config['elastic_net']['enabled']:
            self._elastic_net()

        if self.config['bayesian']['enabled']:
            self._bayesian()

        if self.config['ard']['enabled']:
            self._ard()

    def _all_correlations(self):
        correlations = []
        config = self.config['all_correlations']
        min_correlation = config['min_correlation']
        num_correlations = config['num_correlations']

        for i, field1 in enumerate(self.enabled_fields):
            for field2 in islice(self.enabled_fields, i):
                correlation = self._weighted_correlation(self.df[field1.key], self.df[field2.key])
                if not min_correlation or correlation > min_correlation:
                    correlations.append((field1, field2, correlation))
        correlations.sort(key=lambda t: abs(t[2]), reverse=True)

        if num_correlations:
            correlations = correlations[:num_correlations]
        self._print_correlation_summary(num_correlations=num_correlations, min_correlation=min_correlation)
        for field1, field2, correlation in correlations:
            self._log(f'- {' / '.join(sorted([field1.name, field2.name]))}: {correlation:3f}')

    def _linear(self):
        config = self.config['linear']
        for x_field in self.x_fields:
            correlations = [
                (y_field, self._linear_field(y_field, x_field)) for y_field in self.y_fields if x_field != y_field]
            num_correlations = config['num_correlations']
            min_correlation = config['min_correlation']
            self._print_correlation_summary(field=x_field, num_correlations=num_correlations,
                                            min_correlation=min_correlation)
            self._print_field_weights(correlations, max_count=num_correlations, min_weight=min_correlation)

    def _print_correlation_summary(self, field: Field | None = None, num_correlations: int | None = None,
                                   min_correlation: float | None = None):
        s = []
        if num_correlations:
            s.append(f'Top {num_correlations} correlations')
        else:
            s.append('All correlations')
        if field:
            s.append(f'for {field.name}')
        else:
            s.append('between all enabled fields')
        if min_correlation:
            s.append(f'with magnitude > {min_correlation}')

        self._log(f'\n{' '.join(s)}')

    def _weighted_by(self, include_unweighted: bool = True) -> str | None:
        config = self.config['weighted_by']
        if config['frequency']:
            if config['usage']:
                return 'weighted by inverse frequency and usage'
            return 'weighted by inverse frequency'
        if config['usage']:
            return 'weighted by usage'
        if include_unweighted:
            return 'not weighted'
        return None

    def _print_field_weights(self, fields_and_weights: list[tuple[Field, float]], prefix: str = '- ',
                             max_count: int | None = None, min_weight: float | None = 0):
        fields_and_weights.sort(key=lambda x: abs(x[1]), reverse=True)
        if max_count:
            fields_and_weights = fields_and_weights[:max_count]
        for field, weight in fields_and_weights:
            if not min_weight or abs(weight) > min_weight:
                self._log(f'{prefix}{field.name}: {weight:.3f}')

    # correlations and
    def _linear_field(self, y_field: Field, x_field: Field) -> float:
        config = self.config['linear']
        x = self.df[x_field.key]
        y = self.df[y_field.key]
        correl = self._weighted_correlation(x, y)
        if y_field.plot and x_field.plot and config['plot']:
            # noinspection PyTypeChecker
            x_min: float = x.min()
            # noinspection PyTypeChecker
            x_max: float = x.max()

            polyline = np.linspace(x_min, x_max, 100)

            # linear regression
            poly1 = Polynomial.fit(x, y, 1, w=self.df['Weight'])
            plt.plot(polyline, poly1(polyline), color='blue')

            # quadratic regression
            if config['plot_quadratic']:
                poly2 = Polynomial.fit(x, y, 2, w=self.df['Weight'])
                c, b, a = poly2.convert().coef

                # plot minima or maxima of quadratic regression, if in domain
                x_extrema = -b / (2 * a)
                # noinspection PyUnresolvedReferences
                if x_min <= x_extrema <= x_max:
                    plt.axvline(x_extrema, color='red', linestyle='--', linewidth=1)

                plt.plot(polyline, poly2(polyline), color='red')

            weighted_by = self._weighted_by(include_unweighted=False)
            title = f'{y_field.title} vs. {x_field.title}'
            if weighted_by:
                title += f' - {weighted_by}'

            plt.scatter(x, y)
            plt.xlabel(f'{y_field.name} vs. {x_field.name} (r = {correl:.2f})')
            plt.ylabel(y_field.name)
            plt.title(title)
            plt.tight_layout()
            if config['save_plots']:
                plt.savefig(self._plot_filename(y_field, x_field), bbox_inches='tight')
            plt.show()

        return correl

    def _elastic_net(self):
        config = self.config['elastic_net']
        self._log(f'\nNon-zero ElasticNet weights with alpha {config['alpha']} '
                  f'and l1_ratio = {config['l1_ratio']}:')
        for y_field in self.multi_y_fields:
            model = SGDRegressor(penalty="elasticnet", alpha=config['alpha'],
                                 l1_ratio=config['l1_ratio'], fit_intercept=True,
                                 random_state=config['seed'])
            model.fit(self.multi_x_scaled, self.df[y_field.key], sample_weight=self.df['Weight'])
            self._print_multi_field_weights(y_field, model.coef_)

    def _bayesian(self):
        min_weight = self.config['bayesian']['min_weight'] if self.config['bayesian']['min_weight'] else 0
        self._log(f'\nBayesian Ridge weights with magnitude > {min_weight}:')
        for y_field in self.multi_y_fields:
            model = BayesianRidge()
            model.fit(self.multi_x_scaled, self.df[y_field.key], sample_weight=self.df['Weight'])
            self._print_multi_field_weights(y_field, model.coef_, min_weight)

    def _ard(self):
        min_weight = self.config['ard']['min_weight'] if self.config['ard']['min_weight'] else 0
        self._log(f'\nARD weights with magnitude > {min_weight}:')
        for y_field in self.multi_y_fields:
            weights_sqrt = np.sqrt(np.array(self.df['Weight']))
            X_weighted = self.multi_x_scaled * weights_sqrt[:, np.newaxis]
            y_weighted = self.df[y_field.key] * weights_sqrt
            model = ARDRegression()
            model.fit(X_weighted, y_weighted)
            self._print_multi_field_weights(y_field, model.coef_, min_weight)

    def _print_multi_field_weights(self, field: Field, weights: np.ndarray, min_weight: float = 0):
        field_weights = [(self.multi_x_fields[i], weight) for i, weight in enumerate(weights)
                         if abs(weight) > min_weight]
        if field_weights:
            self._log(f'- {field.name}:')
            self._print_field_weights(field_weights, prefix=' -- ')

    def _plot_filename(self, y_field: Field, x_field: Field):
        y_field_name = y_field.key.lower().replace(' ', '_')
        x_field_name = x_field.key.lower().replace(' ', '_')
        return f'{y_field_name}_{x_field_name}_{self._base_filename()}.png'

    def _log_filename(self):
        return f'results_{self._base_filename()}.txt'

    def _base_filename(self) -> str:
        # noinspection PyStringConversionWithoutDunderMethod
        s = [str(self.df['Date'].max())]
        if self.config['weighted_by']['frequency']:
            s.append('freq')
        if self.config['weighted_by']['usage']:
            s.append('usage')
        return '_'.join(s)

    def _weighted_correlation(self, x, y):
        """Calculates the weighted Pearson correlation coefficient."""
        weights = self.df['Weight']
        # Compute weighted means
        mean_x = np.average(x, weights=weights)
        mean_y = np.average(y, weights=weights)

        # Compute weighted covariances
        cov_xx = np.average((x - mean_x) ** 2, weights=weights)
        cov_yy = np.average((y - mean_y) ** 2, weights=weights)
        cov_xy = np.average((x - mean_x) * (y - mean_y), weights=weights)

        # Compute correlation
        return cov_xy / np.sqrt(cov_xx * cov_yy)

    # noinspection PyTypeChecker
    def _next_pressure(self):
        df = self.df
        avg_pressure = df['Pressure'].mean()
        self._log(f'Mean Pressure: {avg_pressure :.3f}')
        if self.config['filter']['max_days']:
            min_date = df['DateTime'].max() - pd.Timedelta(days=self.config['filter']['max_days'] - 1)
            if df.at[df.index[0], 'DateTime'] == min_date:
                df = df.iloc[1:]
                avg_pressure = df['Pressure'].mean()
                # noinspection PyStringConversionWithoutDunderMethod
                self._log(
                    f'Will drop {df.at[df.index[0], 'Date']} (Pressure {df.at[df.index[0], 'Pressure']}) tomorrow '
                    f'(new mean {avg_pressure:.3f})')

        if self.config['weighted_by']['usage']:
            # weight by usage (same scale as row count)
            avg_usage = df['Usage'].mean()
            pressure_weights = {
                pressure: df[df['Pressure'] == pressure]['Usage'].sum() / avg_usage for pressure in self.valid_pressures
            }
        else:
            # weight by row count
            pressure_weights = df['Pressure'].value_counts()
        target_pressure = np.mean([self.min_pressure, self.max_pressure]) * (len(df) + 1) - df['Pressure'].sum()
        next_pressure = self.min_pressure
        best_score = float('inf')
        config = self.config['next_pressure']

        for pressure in self.valid_pressures:
            pressure_weight = pressure_weights.get(pressure, 0)
            pressure_adjustment = abs(target_pressure - pressure) * config['target_pressure_weight']
            if pressure == self.last_pressure:
                last_pressure_adjustment = config['last_pressure_boost']
            else:
                last_pressure_adjustment = 0
            if config['random_weight']:
                random_adjustment = random.random() * config['random_weight']
            else:
                random_adjustment = 0
            score = pressure_weight + pressure_adjustment - last_pressure_adjustment + random_adjustment
            if score < best_score:
                next_pressure = pressure
                best_score = score

        if next_pressure < self.last_pressure:
            self._log(f'Decrease Pressure from {self.last_pressure} to {next_pressure}')
        elif next_pressure > self.last_pressure:
            self._log(f'Increase Pressure from {self.last_pressure} to {next_pressure}')
        else:
            self._log(f'Leave Pressure at {next_pressure}')


if __name__ == '__main__':
    Regression('config.yaml').run()
