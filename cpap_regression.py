import random
import sys
from dataclasses import dataclass
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from numpy.polynomial.polynomial import Polynomial
from pandas import DatetimeIndex
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
        self.y_fields = [field for field in self.enabled_fields if field.y_field]
        self.x_fields = [field for field in self.enabled_fields if field.x_field]
        self.multi_x_fields = [field for field in self.enabled_fields if field.multi_x_field]
        self.multi_y_fields = [field for field in self.enabled_fields if field.multi_y_field]

        columns = {'Weight', 'Date', 'Pressure'}
        filter_config = self.config['filter']
        if 'max_leak_rate' in filter_config:
            columns.add('AvgLR')
        if 'min_usage' in filter_config or self.config['weighted_by']['usage']:
            columns.add('Usage')
        if 'min_sleep' in filter_config:
            columns.add('Sleep')
        calculate_efficiency = 'min_sleep_efficiency' in filter_config
        calculate_rdi = False

        for field in self.enabled_fields:
            if field.key == 'RDI':
                calculate_rdi = True
            elif field.key == 'Efficiency':
                calculate_efficiency = True
            elif field.key == 'Timestamp':
                pass  # calculated from Date, which is always included
            else:
                columns.add(field.key)
        if calculate_efficiency:
            columns.add('Usage')
            columns.add('Sleep')
        if calculate_rdi:
            columns.add('AHI')
            columns.add('RERA')

        self.df = pd.read_csv(self.config['data_file'], usecols=list(columns))

        self.df['DateTime'] = pd.to_datetime(self.df['Date'], format='%m/%d/%Y')
        self.df['Timestamp'] = self.df['DateTime'].astype('int64') / 1e9
        self.df['Date'] = self.df['DateTime'].dt.strftime('%Y-%m-%d')
        self.df.sort_values(by='DateTime', inplace=True)

        date_counts = self.df['Date'].value_counts()
        if date_counts.iloc[0] > 1:
            print(f'Duplicate Date: {date_counts.index[0]}')
            sys.exit(1)

        self.min_date_time, self.max_date_time, self.num_days = self._filter_dates()

        # Dates have to be known before opening log file (ignoring value filtering)
        self.log_file = open(self._log_filename(), 'w') if self.config['save_logs'] else None

        count = len(self.df)
        # noinspection PyStringConversionWithoutDunderMethod
        date_string = f'{self.min_date_time.strftime('%Y-%m-%d')} and {self.max_date_time.strftime('%Y-%m-%d')}'
        self._log(f'{count} rows between {date_string} ({self.num_days} days)')

        if filter_config['min_date'] and pd.to_datetime(filter_config['min_date']) != self.min_date_time:
            self._log("Config 'min_date' not used")
        if filter_config['max_date'] and pd.to_datetime(filter_config['max_date']) != self.max_date_time:
            self._log("Config 'max_date' not used")

        if count < self.num_days:
            missing = self.num_days - count
            self._log(f'Missing {missing} {'rows' if missing > 1 else 'row'} ({100 * missing / self.num_days:.1f}%)')

        # Pressure field can be empty or contain an exclusion note
        self.df['Pressure'] = pd.to_numeric(self.df['Pressure'], errors='coerce')

        # Weight field is 1 by default (mostly intended for manual exclusion, but non-zero weights also work)
        self.df['Weight'] = self.df['Weight'].fillna(1)

        # drop invalid data (including 0 weight, which is possible with manual weighting)
        self.df.drop(self.df[self.df['Weight'] == 0].index, inplace=True)
        self.df.dropna(inplace=True)
        count = self._print_dropped(count, 'with invalid data')

        if count == 0:
            print('No data')
            sys.exit(1)

        # get last pressure before config filtering so next pressure logic will work correctly
        self.last_pressure = self.df['Pressure'].iloc[-1]

        # convert time fields from H:MM to float hours
        if 'Usage' in columns:
            self.df['Usage'] = pd.to_timedelta(self.df['Usage'] + ':00').dt.total_seconds() / 3600
        if 'Sleep' in columns:
            self.df['Sleep'] = pd.to_timedelta(self.df['Sleep'] + ':00').dt.total_seconds() / 3600
        if 'REM' in columns:
            self.df['REM'] = pd.to_timedelta(self.df['REM'] + ':00').dt.total_seconds() / 3600
        if 'Deep' in columns:
            self.df['Deep'] = pd.to_timedelta(self.df['Deep'] + ':00').dt.total_seconds() / 3600

        # calculated fields
        if calculate_efficiency:
            self.df['Efficiency'] = self.df['Sleep'] / self.df['Usage']

        if calculate_rdi:
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
            self._print_dropped(count, 'for configured filters')

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

    def _print_dropped(self, old_count: int, description: str) -> int:
        new_count = len(self.df)
        dropped = old_count - new_count
        assert dropped >= 0
        if dropped > 0:
            self._log(
                f'Dropped {dropped} {'rows' if dropped > 1 else 'row'} ({100 * dropped / self.num_days:.1f}%) {description}')
        return new_count

    # noinspection PyTypeChecker,PyPackages
    def _filter_dates(self) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        min_date_time = self.df['DateTime'].iloc[0]
        max_date_time = self.df['DateTime'].iloc[-1]

        filter_config = self.config['filter']
        if filter_config['max_date']:
            config_max_date = pd.to_datetime(filter_config['max_date'])
            if config_max_date < max_date_time:
                max_date_time = config_max_date
                self.df.drop(self.df[self.df['DateTime'] > max_date_time].index, inplace=True)
        if filter_config['min_date']:
            config_min_date = pd.to_datetime(filter_config['min_date'])
            if config_min_date > min_date_time:
                min_date_time = config_min_date

        num_days = (max_date_time - min_date_time).days + 1
        if filter_config['max_days']:
            max_days = filter_config['max_days']
            if max_days < num_days:
                num_days = max_days
                min_date_time = max_date_time - pd.Timedelta(days=num_days - 1)

        if min_date_time > self.df['DateTime'].iloc[0]:
            self.df.drop(self.df[self.df['DateTime'] < min_date_time].index, inplace=True)

        return min_date_time, max_date_time, num_days

    def _log(self, s: str):
        print(s)
        if self.log_file:
            print(s, file=self.log_file)

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
                self._log(f'Dropped {num_removed} {'rows' if num_removed > 1 else 'row'} for '
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
        elif num_removed == 0:
            self._log(f"Config '{config_key}' filtered no rows")

        self.df = filtered_df
        return new_dates

    def run(self):
        # noinspection PyStringConversionWithoutDunderMethod
        self._log(f'\nN={len(self.df)} ({100 * len(self.df) / self.num_days:.1f}%) - {self._weighted_by()}')
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
                if not min_correlation or abs(correlation) > min_correlation:
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
        center_pressure: float = np.mean([self.min_pressure, self.max_pressure])

        def mean_pressure(avg_pressure: float) -> str:
            if avg_pressure > center_pressure:
                suffix = f' ({avg_pressure - center_pressure:.3f} above center)'
            elif avg_pressure < center_pressure:
                suffix = f' ({center_pressure - avg_pressure:.3f} below center)'
            else:
                suffix = ''

            return f'{avg_pressure:.3f}{suffix}'

        self._log(f'Mean Pressure: {mean_pressure(avg_pressure)}')
        dropped_pressure = None
        if self.config['filter']['max_days'] and df.at[df.index[0], 'DateTime'] == self.min_date_time:
            df = df.iloc[1:]
            avg_pressure = df['Pressure'].mean()
            # noinspection PyStringConversionWithoutDunderMethod
            dropped_pressure = df.at[df.index[0], 'Pressure']
            self._log(f'Will drop {df.at[df.index[0], 'Date']} (Pressure {dropped_pressure}) tomorrow')
            self._log(f'  New mean Pressure: {mean_pressure(avg_pressure)}')

        if self.config['weighted_by']['usage']:
            # weight by usage (same scale as row count)
            avg_usage = df['Usage'].mean()
            pressure_weights = {
                pressure: df[df['Pressure'] == pressure]['Usage'].sum() / avg_usage for pressure in self.valid_pressures
            }
        else:
            # weight by row count
            pressure_weights = df['Pressure'].value_counts()

        next_pressure = self.max_pressure
        lowest_score = float('inf')
        extreme_pressures = (self.min_pressure, self.max_pressure)
        if dropped_pressure is not None and pressure_weights.get(dropped_pressure, 0) == 0:
            if dropped_pressure in extreme_pressures:
                # always select dropped pressure if it reduces count at either extreme to 0
                # otherwise the pressure will be lost unless min_pressure or max_pressure is set
                # there is a similar check in the for loop, but prioritize dropped pressure
                # since the other extreme must be locked by config
                next_pressure = dropped_pressure
                lowest_score = -float('inf')

        config = self.config['next_pressure']

        if config['verbose']:
            self._log('Scores:')
        for pressure in self.valid_pressures:
            pressure_weight = pressure_weights.get(pressure, 0)

            if pressure_weight == 0 and pressure in extreme_pressures:
                # always select extreme weight with zero count (so it isn't lost or manual override can be removed)
                pressure_boost = float('inf')
            elif config['last_pressure_boost'] and pressure == self.last_pressure:
                # otherwise prefer most recent pressure
                pressure_boost = config['last_pressure_boost']
            else:
                pressure_boost = 0

            if config['random_weight']:
                random_adjustment = random.random() * config['random_weight']
            else:
                random_adjustment = 0

            score = pressure_weight + random_adjustment - pressure_boost
            if config['verbose']:
                self._log(f'- {pressure}: {score:.2f}'
                          f' ({pressure_weight:.2f} + {random_adjustment:.2f} - {pressure_boost})')
            if score < lowest_score:
                next_pressure = pressure
                lowest_score = score

        if next_pressure < self.last_pressure:
            self._log(f'Decrease Pressure from {self.last_pressure} to {next_pressure}')
        elif next_pressure > self.last_pressure:
            self._log(f'Increase Pressure from {self.last_pressure} to {next_pressure}')
        else:
            self._log(f'Leave Pressure at {next_pressure}')


if __name__ == '__main__':
    Regression('config.yaml').run()
