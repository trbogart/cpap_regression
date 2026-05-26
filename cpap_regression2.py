import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    include_leak_rate = False
    alpha = 1.0
    min_pressure = 7.0
    max_leak_rate = 1.2
    min_usage = 5

    df = pd.read_csv('cpap.csv')
    df['Usage'] = pd.to_timedelta(df['Usage'] + ':00').dt.total_seconds() / 3600
    if min_pressure is not None:
        df = df[df['Pressure'] >= min_pressure]
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
