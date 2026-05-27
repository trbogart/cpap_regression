import pandas as pd
import yaml
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        # Use safe_load to avoid executing arbitrary code from the file
        config = yaml.safe_load(file)


    def config_float(key):
        s = config.get(key)
        return float(s) if s is not None else None


    include_leak_rate = False
    alpha = config_float('alpha')
    min_pressure = config_float('min_pressure')
    max_pressure = config_float('max_pressure')
    max_leak_rate = config_float('max_leak_rate')
    min_usage = config_float('min_usage')

    df = pd.read_csv('cpap.csv')
    df['Usage'] = pd.to_timedelta(df['Usage'] + ':00').dt.total_seconds() / 3600
    if min_pressure is not None:
        df = df[df['Pressure'] >= min_pressure]
    if max_pressure is not None:
        df = df[df['Pressure'] >= max_pressure]
    if max_leak_rate is not None:
        df = df[df['AvgLR'] <= max_leak_rate]
    if min_usage is not None:
        df = df[df['Usage'] >= min_usage]

    X = df[['Pressure', 'AvgLR']] if include_leak_rate else df[['Pressure']]
    X = StandardScaler().fit_transform(X)

    y_fields = {
        'Comb FL': 'Combined FL (WAT/NED)',
        'FLS': 'Flow Limitation Score (WAT)',
        'GI': 'Glasgow Index',
        'NED RDI': 'Est. RDI (NED)',
        'BOI': 'Brief Obstruction Index',

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

    for key, value in y_fields.items():
        y = df[[key]]
        model = ElasticNet(alpha=alpha, l1_ratio=1, fit_intercept=True)
        model.fit(X, y)
        if sum(model.coef_) > 0:
            print(f'{key}: {model.intercept_} + {model.coef_}')
