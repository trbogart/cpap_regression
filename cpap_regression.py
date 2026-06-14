import argparse
import random
import sys
from dataclasses import dataclass
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from numpy.polynomial.polynomial import Polynomial
from pandas import DatetimeIndex
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import ARDRegression, BayesianRidge, SGDRegressor, LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

default_config_file = 'config.yaml'


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
        return cls(key, name, title, plot, y_field, x_field, multi_x_field)


class Regression:
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

        columns = {'Date', 'Pressure'}
        filter_config = self.config['filter']
        if 'max_leak_rate' in filter_config:
            columns.add('AvgLR')
        if 'min_usage' in filter_config:
            columns.add('Usage')
        if 'min_sleep' in filter_config:
            columns.add('Sleep')
        if 'collar' in filter_config:
            columns.add('Collar')
        calculate_efficiency = 'min_sleep_efficiency' in filter_config
        calculate_rdi = False
        calculate_ned_mean_split = False

        for field in self.enabled_fields:
            if field.key == 'RDI':
                calculate_rdi = True
            elif field.key == 'Efficiency':
                calculate_efficiency = True
            elif field.key == 'Timestamp' or field.key == 'DateTime':
                pass  # calculated from Date, which is always included
            elif field.key == 'NED Mean Split':
                calculate_ned_mean_split = True
            else:
                columns.add(field.key)
        if calculate_efficiency:
            columns.add('Usage')
            columns.add('Sleep')
        if calculate_rdi:
            columns.add('AHI')
            columns.add('RERA')
        if calculate_ned_mean_split:
            columns.add('H1 NED Mean')
            columns.add('H2 NED Mean')

        self.df = pd.read_csv(self.config['data_file'], usecols=list(columns))

        self.df['DateTime'] = pd.to_datetime(self.df['Date'], format='%m/%d/%Y')
        self.df['Timestamp'] = self.df['DateTime'].astype('int64') / 1e9
        self.df['Date'] = self.df['DateTime'].dt.strftime('%Y-%m-%d')
        self.df.sort_values(by='DateTime', inplace=True)
        self.min_date_time, self.max_date_time, self.num_days = self._filter_dates()

        self.tags = [self.config['tag']] if self.config['tag'] else []
        if self.config['stats']['bucket']:
            self.tags.append('bucket')

        # Dates have to be known before opening log file (ignoring value filtering)
        if self.config['stats']['enabled'] and self.config['save_logs']:
            self.log_file = open(self._log_filename(), 'w')
        else:
            self.log_file = None

        date_counts = self.df['Date'].value_counts()
        if date_counts.iloc[0] > 1:
            self._log(f'Duplicate Date: {date_counts.index[0]}')
            sys.exit(1)

        count = len(self.df)
        # noinspection PyStringConversionWithoutDunderMethod
        date_string = f'{self.min_date_time.strftime('%Y-%m-%d')} and {self.max_date_time.strftime('%Y-%m-%d')}'
        self._log(f'{count} rows between {date_string} ({self.num_days} days)')

        if filter_config['min_date'] and pd.to_datetime(filter_config['min_date']) != self.min_date_time:
            self._log("Config 'min_date' can be edited")
        if filter_config['max_date'] and pd.to_datetime(filter_config['max_date']) != self.max_date_time:
            self._log("Config 'max_date' can be edited")

        if count < self.num_days:
            missing = self.num_days - count
            self._log(f'Missing {missing} {'rows' if missing > 1 else 'row'} ({100 * missing / self.num_days:.1f}%)')

        # Pressure field can be empty or contain an exclusion note
        self.df['Pressure'] = pd.to_numeric(self.df['Pressure'], errors='coerce')

        # Collar field is 0 by default
        if 'Collar' in columns:
            self.df['Collar'] = self.df['Collar'].fillna(0)

        # drop invalid data
        self.df.dropna(inplace=True)
        count = self._print_dropped(count, 'with invalid data')

        if count == 0:
            self._log('No data')
            sys.exit(1)

        pressure_transform: dict[float, float] = self.config['pressure_transform']
        if pressure_transform:
            pressure_counts = self.df['Pressure'].value_counts()
            for pressure in pressure_transform.keys():
                if pressure_counts.get(pressure, 0) == 0:
                    self._log(f"Config 'pressure_transform[{pressure:.1f}]' is not used")
            self.df['Pressure'] = self.df['Pressure'].replace(pressure_transform)

        # get last pressure before config filtering so next pressure logic will work correctly
        self.last_pressure: float = self.df['Pressure'].iloc[-1]

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

        if calculate_ned_mean_split:
            self.df['NED Mean Split'] = np.abs(self.df['H2 NED Mean'] - self.df['H1 NED Mean'])

        # filter by config
        dates = set(self.df['Date'])
        dates = self._filter_config(dates, 'Pressure', 'min_pressure')
        dates = self._filter_config(dates, 'Pressure', 'max_pressure')
        dates = self._filter_config(dates, 'AvgLR', 'max_leak_rate')
        dates = self._filter_config(dates, 'Usage', 'min_usage')
        dates = self._filter_config(dates, 'Sleep', 'min_sleep')
        dates = self._filter_config(dates, 'Efficiency', 'min_sleep_efficiency')
        dates = self._filter_config(dates, 'Collar', 'collar')
        if not self.config['filter']['verbose'] and len(dates) < count:
            self._print_dropped(count, 'for configured filters')
        if len(self.df) == 0:
            self._log('All data filtered')
            sys.exit(1)

        if filter_config['min_pressure']:
            self.min_pressure: float = filter_config['min_pressure']
            if not self._is_pressure_valid(self.min_pressure):
                self._log(f"Invalid 'min_pressure': {self.min_pressure:.1f}")
                sys.exit(1)
            if self.last_pressure < self.min_pressure:
                last_valid_pressure = self.df['Pressure'].iloc[-1]
                if self.config['next_pressure']['enabled']:
                    self._log(f"Last pressure ({self.last_pressure:.1f}) below 'min_pressure' "
                              f"({self.min_pressure:.1f}), using {last_valid_pressure:.1f} instead")
                self.last_pressure = last_valid_pressure
        else:
            # noinspection PyTypeChecker
            self.min_pressure: float = min(self.last_pressure, self.df['Pressure'].min())
        if filter_config['max_pressure']:
            self.max_pressure: float = filter_config['max_pressure']
            if not self._is_pressure_valid(self.max_pressure):
                self._log(f"Invalid 'max_pressure': {self.max_pressure:.1f}")
                sys.exit(1)
            if self.last_pressure > self.max_pressure:
                last_valid_pressure = self.df['Pressure'].iloc[-1]
                if self.config['next_pressure']['enabled']:
                    self._log(
                        f"Last pressure ({self.last_pressure:.1f}) above 'max_pressure' "
                        f"({self.max_pressure:.1f}), using {last_valid_pressure:.1f} instead")
                self.last_pressure = last_valid_pressure
        else:
            # noinspection PyTypeChecker
            self.max_pressure: float = max(self.last_pressure, self.df['Pressure'].max())

        self.valid_pressures = [p / 5 for p in range(int(round(self.min_pressure * 5)),
                                                     int(round(self.max_pressure * 5)) + 1)]

        num_pressures = len(self.valid_pressures)
        if self.config['stats']['bucket'] and num_pressures > 3:
            if num_pressures % 2 == 1:
                self._log("Warning: 'bucket' used with odd number of buckets")
            pressure_bucket_map = {}
            new_valid_pressures = []

            last_bucket_pressure = 0
            for i, pressure in enumerate(self.valid_pressures):
                if i % 2 == 0:
                    last_bucket_pressure = round(pressure + 0.1, 1)
                    new_valid_pressures.append(last_bucket_pressure)
                    pressure_bucket_map[pressure] = last_bucket_pressure
                else:
                    pressure_bucket_map[pressure] = last_bucket_pressure

            self.df['Pressure'] = self.df['Pressure'].replace(pressure_bucket_map)
            self.valid_pressures = new_valid_pressures
            self.min_pressure = pressure_bucket_map[self.min_pressure]
            self.max_pressure = pressure_bucket_map[self.max_pressure]
            self.last_pressure = pressure_bucket_map[self.last_pressure]

        self.center_pressure = (self.min_pressure + self.max_pressure) / 2

        self.multi_x_scaled = StandardScaler().fit_transform(self.df[[field.key for field in self.multi_x_fields]])

        self.dropped_date: DatetimeIndex | None = None
        self.dropped_pressure: float | None = None
        self.df_tomorrow = self.df
        if self.config['filter']['max_days'] and self.num_days == self.config['filter']['max_days']:
            # at maximum number of days, but first day may be missing or invalid
            # note: can't just check self.min_date_time first because it can also be set by min_date config
            if self.df.at[self.df.index[0], 'DateTime'] == self.min_date_time:
                # noinspection PyTypeChecker
                self.dropped_pressure = self.df.at[self.df.index[0], 'Pressure']
                # noinspection PyTypeChecker
                self.dropped_date: str = self.df.at[self.df.index[0], 'Date']
                self.df_tomorrow = self.df.iloc[1:]

    def run(self):
        # noinspection PyStringConversionWithoutDunderMethod
        self._log(f'\nN={len(self.df)} ({100 * len(self.df) / self.num_days:.1f}%)')

        if self.config['pressure_counts']['enabled']:
            self._pressure_counts()
        else:
            # noinspection PyStringConversionWithoutDunderMethod
            date_string = f'{self.df.at[self.df.index[0], 'Date']} and {self.df.at[self.df.index[-1], 'Date']}'
            self._log(f'Valid data between {date_string}')
            if self.dropped_pressure is not None:
                # noinspection PyStringConversionWithoutDunderMethod
                self._log(f'Will drop {self.dropped_date} (Pressure {self.dropped_pressure:.1f}) tomorrow')
            else:
                self._log('Will not drop a row tomorrow')

        if len(self.df) < 2:
            self._log('Minimum N=2')
            sys.exit(0)

        if self.config['stats']['enabled']:
            if self.config['all_correlations']['enabled']:
                self._all_correlations()

            # Correlations, linear, and quadratic regressions
            self._linear_quadratic()

            # Elastic Net to regularize multi_x_fields
            self._elastic_net()

            # Bayesian Ridge to regularize multi_x_fields
            self._bayesian_ridge()

            # ARD Regression to regularize multi_x_fields
            self._ard()

            # Linear partial dependence
            self._linear_partial_dependence()

        if self.config['next_pressure']['enabled']:
            self._next_pressure()

    def _pressure_counts(self):
        self._log('Pressure Counts:')
        for pressure in self.valid_pressures:
            data_for_pressure = self.df[self.df['Pressure'] == pressure]
            dates = data_for_pressure['Date']
            max_dates = self.config['pressure_counts']['max_dates']
            if max_dates and len(dates) > max_dates:
                half = (max_dates - 1) // 2
                dates_str = list(dates)[:half]
                dates_str.append('...')
                dates_str.extend(dates[-half:])
            else:
                dates_str = dates

            total_usage = data_for_pressure['Usage'].sum()
            self._log(f'- {pressure:.1f} ({len(dates)} count, {total_usage:.1f} hrs): {', '.join(dates_str)}')

        def print_summary(df, prefix: str = ''):
            avg_pressure = df['Pressure'].mean()
            self._log(f'{prefix}Mean Pressure: {self._mean_pressure_string(avg_pressure)}')

            if self.config['pressure_counts']['pressure_date_correlation']:
                correl = np.corrcoef(df['Pressure'], df['Timestamp'])[0, 1]
                self._log(f'{prefix}Correlation between Pressure and Date: {self._get_correlation_string(correl)}')

        print_summary(self.df)

        if self.dropped_pressure is not None:
            # noinspection PyStringConversionWithoutDunderMethod
            self._log(f'Will drop {self.dropped_date} (Pressure {self.dropped_pressure:.1f}) tomorrow')
            print_summary(self.df_tomorrow, '- ')
        else:
            self._log('Will not drop a row tomorrow')

    def _mean_pressure_string(self, avg_pressure: float) -> str:
        center_diff = avg_pressure - self.center_pressure
        if center_diff >= 0:
            suffix = f' ({center_diff :.3f} above center: {self.center_pressure:.1f})'
        else:
            suffix = f' ({-center_diff:.3f} below center: {self.center_pressure:.1f})'

        return f'{avg_pressure:.3f}{suffix}'

    def _next_pressure(self):
        # calculate next pressure
        # priority is:
        # 1 - last pressure if either min or max and count is 0 (implying data was invalid, to avoid pressure "falling off")
        # 2 - dropped pressure if either min or max and new count will be 0 (to avoid pressure "falling off")
        # 3 - min pressure if count is 0 (will be able to remove min_pressure config)
        # 4 - max pressure if count is 0 (will be able to remove max_pressure config)
        # 5 - select lowest adjusted weight
        #     - base weight is count (or total usage scaled to 1.0/night average if weighted by usage)
        #     - subtract last_pressure_boost for most recent pressure
        #     - add random Gaussian number with random_sigma
        #     - add distance from pressure that would center pressure multiplied by center_weight

        df = self.df_tomorrow

        pressure_counts = df['Pressure'].value_counts()
        extreme_pressures = {self.min_pressure, self.max_pressure}

        def is_zero_extreme(pr: float) -> bool:
            return pr in extreme_pressures and pressure_counts.get(pr, 0) == 0

        # extreme pressure with zero count will always be prioritized, but last pressure or dropped pressure
        # may not be locked with min_pressure or max_pressure config (only matters if both extreme counts are zero)
        next_pressure = best_score = float('inf')
        if is_zero_extreme(self.last_pressure):
            next_pressure = self.last_pressure
            best_score = float('-inf')
        elif self.dropped_pressure is not None and is_zero_extreme(self.dropped_pressure):
            next_pressure = self.dropped_pressure
            best_score = float('-inf')

        # pressure that will make mean pressure equal to center pressure
        target_pressure = round(self.center_pressure * (len(df) + 1) - df['Pressure'].sum(), 1)

        if self.config['next_pressure']['verbose']:
            self._log(
                f'\nPressure that would move mean Pressure to center ({self.center_pressure:.1f}): {target_pressure:.1f}')
            self._log(f'Next Pressure Scores:')

        last_pressure_boost = self.config['next_pressure']['last_pressure_boost']
        pressure_boosts = self.config['next_pressure']['pressure_boosts']
        center_weight = self.config['next_pressure']['center_weight']
        random_sigma = self.config['next_pressure']['random_sigma']

        for pressure in self.valid_pressures:
            pressure_count = pressure_counts.get(pressure, 0)

            if pressure_count == 0 and pressure in extreme_pressures:
                # always select extreme pressure with zero count
                pressure_boost = float('inf')
            elif last_pressure_boost and pressure == self.last_pressure:
                # otherwise prefer most recent pressure
                pressure_boost = last_pressure_boost
            else:
                pressure_boost = 0

            if pressure_boosts:
                pressure_boost += pressure_boosts.get(pressure, 0)

            if center_weight:
                center_distance = round(abs(pressure - target_pressure) * center_weight, 1)
            else:
                center_distance = 0

            if random_sigma:
                random_adjustment = random.gauss(sigma=random_sigma)
            else:
                random_adjustment = 0

            score = pressure_count + random_adjustment + center_distance - pressure_boost
            if self.config['next_pressure']['verbose']:
                self._log(f'- {pressure:3.1f}: {score:5.2f} = '
                          f'{pressure_count:2d} count '
                          f'{random_adjustment:+.2f} random '
                          f'{center_distance:+g} center '
                          f'{-pressure_boost:+g} boost')
            if score < best_score:
                next_pressure = pressure
                best_score = score

        # noinspection PyTypeChecker,PyUnresolvedReferences
        tomorrow = (self.max_date_time + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        if next_pressure < self.last_pressure:
            self._log(f'\nDecrease Pressure from {self.last_pressure:.1f} to {next_pressure:.1f} for {tomorrow}')
        elif next_pressure > self.last_pressure:
            self._log(f'\nIncrease Pressure from {self.last_pressure:.1f} to {next_pressure:.1f} for {tomorrow}')
        else:
            self._log(f'\nLeave Pressure at {next_pressure:.1f} for {tomorrow}')

    def _print_dropped(self, old_count: int, description: str) -> int:
        new_count = len(self.df)
        dropped = old_count - new_count
        assert dropped >= 0
        if dropped > 0:
            self._log(
                f'Dropped {dropped} {'rows' if dropped > 1 else 'row'} ({100 * dropped / self.num_days:.1f}%) {description}')
        return new_count

    @staticmethod
    def _is_pressure_valid(pressure: float) -> bool:
        scaled = pressure * 5
        return round(scaled) == scaled

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

    def _filter_config(self, dates: set, field_name: str, config_key: str) -> set:
        threshold = self.config['filter'][config_key]
        if threshold is None:
            return dates
        if config_key.startswith('min_'):
            filtered_df = self.df[self.df[field_name] >= threshold]
        elif config_key.startswith('max_'):
            filtered_df = self.df[self.df[field_name] <= threshold]
        else:
            filtered_df = self.df[self.df[field_name] == threshold]

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
                self._log(f'Dropped {num_removed} rows with zero {field_name}')
            if num_removed > 0:
                for _, row in removed_rows.iterrows():
                    line = f'- {row['Date']}: Pressure={row['Pressure']:.1f}'
                    if field_name != 'Pressure':
                        line += f', {field_name}={row[field_name]:.2f}'
                    self._log(line)
        elif num_removed == 0 and field_name == 'Pressure':
            # only display this for controllable fields
            self._log(f"Config '{config_key}' can be edited")

        self.df = filtered_df
        return new_dates

    def _all_correlations(self):
        correlations = []
        config = self.config['all_correlations']
        min_correlation = config['min_correlation']
        num_correlations = config['num_correlations']

        for i, field1 in enumerate(self.enabled_fields):
            for field2 in islice(self.enabled_fields, i):
                correlation = np.corrcoef(self.df[field1.key], self.df[field2.key])[0, 1]
                if not min_correlation or abs(correlation) >= min_correlation:
                    correlations.append((field1, field2, correlation))
        correlations.sort(key=lambda t: abs(t[2]), reverse=True)

        if num_correlations:
            correlations = correlations[:num_correlations]
        self._print_correlation_summary(num_correlations=num_correlations, min_correlation=min_correlation)
        for field1, field2, correlation in correlations:
            self._log(
                f'- {' / '.join(sorted([field1.name, field2.name]))}: {self._get_correlation_string(correlation)}')

    def _linear_quadratic(self):
        num_correlations = self.config['correlation']['num_correlations']
        min_correlation = self.config['correlation']['min_correlation']

        for x_field in self.x_fields:
            if len(self.df[x_field.key].unique()) < 2:
                self._log(f'\nSkip {x_field.name}: only 1 value')
            else:
                field_corr_and_r2_scores = [(y_field, self._linear_quadratic_field(y_field, x_field))
                                            for y_field in self.y_fields if x_field != y_field]
                if self.config['correlation']['enabled']:
                    fields_and_correlations = [(y_field, corr) for y_field, (corr, _, _) in field_corr_and_r2_scores]
                    fields_and_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

                    self._print_correlation_summary(field=x_field,
                                                    num_correlations=num_correlations,
                                                    min_correlation=min_correlation)
                    if num_correlations:
                        fields_and_correlations = fields_and_correlations[:num_correlations]
                    for field, correlation in fields_and_correlations:
                        if not min_correlation or abs(correlation) >= min_correlation:
                            self._log(f'- {field.name}: {self._get_correlation_string(correlation)}')

                r2_prefix = 'Adjusted ' if self.config['r2']['adjusted'] else ''

                if self.config['r2']['linear']:
                    r2_scores = [(y_field, r2_linear) for y_field, (_, r2_linear, _) in field_corr_and_r2_scores]
                    self._print_r2(f'{r2_prefix}Linear', x_field, r2_scores)
                if self.config['r2']['quadratic']:
                    r2_scores = [(y_field, r2_quadratic) for y_field, (_, _, r2_quadratic) in field_corr_and_r2_scores]
                    regression_type = f'{r2_prefix}Quadratic'

                    self._print_r2(regression_type, x_field, r2_scores)

    def _print_correlation_summary(self,
                                   field: Field | None = None,
                                   num_correlations: int | None = None,
                                   min_correlation: float | None = None):
        s = []
        if num_correlations:
            s.append(f'Top {num_correlations} correlations')
        else:
            s.append('Correlations')
        if field:
            s.append(f'for {field.name}')
        else:
            s.append('between all enabled fields')  # for all_correlations
        if min_correlation:
            s.append(f'with magnitude > {min_correlation}')

        self._log(f'\n{' '.join(s)}')

    def _print_r2(self, regression_type: str, field: Field, r2_scores: list[tuple[Field, float]]):
        num_scores = self.config['r2']['num_scores']
        min_score = self.config['r2']['min_score']

        s = []
        if num_scores:
            s.append(f'Top {num_scores}')
        s.append(regression_type)
        s.append(f'R² scores')
        if min_score:
            s.append(f'> {min_score}')
            s.append(f'for {field.name}')

        self._log(f'\n{' '.join(s)}')
        self._print_field_weights(r2_scores, max_count=num_scores, min_weight=min_score, sort_by_magnitude = False)

    def _print_field_weights(self, fields_and_weights: list[tuple[Field, float]], prefix: str = '- ',
                             max_count: int | None = None, min_weight: float | None = 0, sort_by_magnitude: bool = True):
        if sort_by_magnitude:
            fields_and_weights.sort(key=lambda x: abs(x[1]), reverse=True)
        else:
            fields_and_weights.sort(key=lambda x: x[1], reverse=True)
        if max_count:
            fields_and_weights = fields_and_weights[:max_count]
        for field, weight in fields_and_weights:
            if not min_weight or abs(weight) >= min_weight:
                self._log(f'{prefix}{field.name}: {weight:.3f}')

    def _r2_score(self, y, y_pred, k: int) -> float:
        r2 = r2_score(y, y_pred)
        if self.config['r2']['adjusted']:
            n = len(self.df)
            return 1 - (1 - r2) * (n - 1) / (n - k - 1)
        return r2

    # return correlation, linear R2 score, and quadratic R2 score
    def _linear_quadratic_field(self, y_field: Field, x_field: Field) -> tuple[float, float, float]:
        x = self.df[x_field.key]
        y = self.df[y_field.key]

        r = np.corrcoef(x, y)[0, 1]

        poly1 = Polynomial.fit(x, y, 1)
        r2_linear = self._r2_score(y, poly1(x), 1)

        poly2 = Polynomial.fit(x, y, 2)
        r2_quadratic =  self._r2_score(y, poly2(x), 1)

        plot_config = self.config['plot']
        if y_field.plot and x_field.plot and plot_config['enabled']:
            if not plot_config['min_r2'] or r2_linear >= plot_config['min_r2'] or r2_quadratic >= plot_config['min_r2']:
                def show_plot(tags: list[str] | None = None):
                    title_lines = [f'{y_field.title} vs. {x_field.title}']
                    r2_prefix = 'adjusted ' if self.config['r2']['adjusted'] else ''
                    title_lines.append(f'{r2_prefix}linear R² = {r2_linear:.3f}, '
                                       f'{r2_prefix}quadratic R² = {r2_quadratic:.3f}')

                    plt.xlabel(f'{x_field.name}')
                    plt.ylabel(y_field.name)
                    plt.title('\n'.join(title_lines))
                    plt.tight_layout()
                    if plot_config['save']:
                        plt.savefig(self._plot_filename(y_field, x_field, tags), bbox_inches='tight')
                    plt.show()

                if plot_config['violin']:
                    sns.violinplot(data=self.df, x=x_field.key, y=y_field.key, inner='quart', density_norm='count')
                    show_plot(tags=['violin'])

                if plot_config['box']:
                    sns.boxplot(data=self.df, x=x_field.key, y=y_field.key)
                    show_plot(tags=['box'])

                if plot_config['linear'] or plot_config['quadratic']:
                    plt.scatter(x, y)

                    polyline = np.linspace(x.min(), x.max(), 100)
                    if plot_config['linear']:
                        # linear regression
                        plt.plot(polyline, poly1(polyline), color='blue')

                    # quadratic regression
                    if plot_config['quadratic']:
                        c, b, a = poly2.convert().coef

                        # plot minima or maxima of quadratic regression, if in domain
                        x_extrema = -b / (2 * a)
                        # noinspection PyUnresolvedReferences
                        if x.min() <= x_extrema <= x.max():
                            plt.axvline(x_extrema, color='red', linestyle='--', linewidth=1)

                        plt.plot(polyline, poly2(polyline), color='red')

                    show_plot()

        return r, r2_linear, r2_quadratic

    def _linear_partial_dependence(self):
        config = self.config['linear_partial_dependence']
        if config['enabled']:
            for y_field in self.y_fields:
                if y_field.plot:
                    model = LinearRegression().fit(self.multi_x_scaled, self.df[y_field.key])
                    fig, axes = plt.subplots(nrows=1, ncols=len(self.multi_x_fields))
                    PartialDependenceDisplay.from_estimator(estimator=model,
                                                            X=self.multi_x_scaled,
                                                            features=list(range(len(self.multi_x_fields))),
                                                            feature_names=[field.name for field in self.multi_x_fields],
                                                            ax=axes)
                    fig.suptitle(y_field.title)
                    plt.tight_layout()
                    if config['save']:
                        filename = f'pd_{y_field.key.replace(' ', '_').lower()}_{self._base_filename()}.png'
                        plt.savefig(filename)
                    plt.show()

    def _elastic_net(self):
        config = self.config['elastic_net']
        if config['enabled']:
            self._log(f'\nNon-zero ElasticNet weights with alpha {config['alpha']} '
                      f'and l1_ratio = {config['l1_ratio']}:')
            for y_field in self.multi_y_fields:
                model = SGDRegressor(penalty="elasticnet", alpha=config['alpha'],
                                     l1_ratio=config['l1_ratio'], fit_intercept=True,
                                     random_state=config['seed'])
                model.fit(self.multi_x_scaled, self.df[y_field.key])
                self._print_multi_field_weights(y_field, model.coef_)

    def _bayesian_ridge(self):
        config = self.config['bayesian_ridge']
        if config['enabled']:
            min_weight = config['min_weight'] if config['min_weight'] else 0
            self._log(f'\nBayesian Ridge weights with magnitude > {min_weight}:')
            for y_field in self.multi_y_fields:
                model = BayesianRidge()
                model.fit(self.multi_x_scaled, self.df[y_field.key])
                self._print_multi_field_weights(y_field, model.coef_, min_weight)

    def _ard(self):
        config = self.config['ard']
        if config['enabled']:
            min_weight = config['min_weight'] if config['min_weight'] else 0
            self._log(f'\nARD weights with magnitude > {min_weight}:')
            for y_field in self.multi_y_fields:
                model = ARDRegression()
                model.fit(self.multi_x_scaled, self.df[y_field.key])
                self._print_multi_field_weights(y_field, model.coef_, min_weight)

    def _print_multi_field_weights(self, field: Field, weights: np.ndarray, min_weight: float = 0):
        field_weights = [(self.multi_x_fields[i], weight) for i, weight in enumerate(weights)
                         if abs(weight) >= min_weight]
        if field_weights:
            self._log(f'- {field.name}:')
            self._print_field_weights(field_weights, prefix=' -- ')

    def _plot_filename(self, y_field: Field, x_field: Field, tags: list[str] | None = None):
        y_field_name = y_field.key.lower().replace(' ', '_')
        x_field_name = x_field.key.lower().replace(' ', '_')
        return f'{y_field_name}_{x_field_name}_{self._base_filename(tags)}.png'

    def _log_filename(self):
        return f'results_{self._base_filename()}.txt'

    def _base_filename(self, extra_tags: list[str] | None = None) -> str:
        tags = self.tags[:]
        if extra_tags:
            tags.extend(extra_tags)
        # noinspection PyStringConversionWithoutDunderMethod
        tags.append(str(self.df['Date'].max()))
        return '_'.join(tags)

    @staticmethod
    def _get_correlation_string(r: float) -> str:
        magnitude = abs(r)
        if magnitude >= 0.8:
            s = 'very strong'
        elif magnitude >= 0.6:
            s = 'strong'
        elif magnitude >= 0.4:
            s = 'moderate'
        elif magnitude >= 0.2:
            s = 'weak'
        else:
            s = 'very weak'
        return f'{r:.2f} ({s})'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=default_config_file,
                        help=f'Configuration file (default {default_config_file})')
    args = parser.parse_args()
    Regression(args.config).run()
