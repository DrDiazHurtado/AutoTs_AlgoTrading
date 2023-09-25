# This is an example of ensemble time series models using AutoTS

from autots import AutoTS
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_squared_error

# These are all the models that can be used with AutoTS
model_dict = {
    'all': [
        'ConstantNaive', 'LastValueNaive', 'AverageValueNaive', 'GLS', 'GLM', 'ETS', 'ARIMA', 'FBProphet', 'RollingRegression', 'GluonTS',
        'SeasonalNaive', 'UnobservedComponents', 'VECM', 'DynamicFactor', 'MotifSimulation', 'WindowRegression', 'VAR', 'DatepartRegression',
        'UnivariateRegression', 'UnivariateMotif', 'MultivariateMotif', 'NVAR', 'MultivariateRegression', 'SectionalMotif', 'Theta', 'ARDL',
        'NeuralProphet', 'DynamicFactorMQ', 'PytorchForecasting', 'ARCH', 'RRVAR', 'MAR', 'TMF', 'LATC', 'KalmanStateSpace', 'MetricMotif',
        'Cassandra', 'SeasonalityMotif', 'MLEnsemble'
    ],
    'default': {
        'ConstantNaive': 1, 'LastValueNaive': 1, 'AverageValueNaive': 1, 'GLS': 1, 'SeasonalNaive': 1, 'GLM': 1, 'ETS': 1, 'FBProphet': 0.5,
        'GluonTS': 0.5, 'UnobservedComponents': 1, 'VAR': 1, 'VECM': 1, 'ARIMA': 0.4, 'WindowRegression': 0.5, 'DatepartRegression': 1,
        'UnivariateRegression': 0.3, 'MultivariateRegression': 0.4, 'UnivariateMotif': 1, 'MultivariateMotif': 1, 'SectionalMotif': 1,
        'NVAR': 1, 'Theta': 1, 'ARDL': 1, 'ARCH': 1, 'MetricMotif': 1
    },
    'fast': {
        'ConstantNaive': 1, 'LastValueNaive': 1.5, 'AverageValueNaive': 1, 'GLS': 1, 'SeasonalNaive': 1, 'GLM': 1, 'ETS': 1, 'VAR': 0.8,
        'VECM': 1, 'WindowRegression': 0.5, 'DatepartRegression': 0.8, 'UnivariateMotif': 1, 'MultivariateMotif': 0.8, 'SectionalMotif': 1,
        'NVAR': 1, 'MAR': 1, 'RRVAR': 1, 'KalmanStateSpace': 1, 'MetricMotif': 1, 'Cassandra': 1, 'SeasonalityMotif': 1
    },
    'superfast': [
        'ConstantNaive', 'LastValueNaive', 'AverageValueNaive', 'GLS', 'SeasonalNaive', 'SeasonalityMotif'
    ],
    'parallel': {
        'ETS': 1, 'FBProphet': 0.8, 'ARIMA': 0.2, 'GLM': 1, 'UnobservedComponents': 1, 'UnivariateMotif': 1, 'MultivariateMotif': 1,
        'Theta': 1, 'ARDL': 1, 'ARCH': 1
    },
    'fast_parallel': {
        'ETS': 1, 'FBProphet': 0.8, 'ARIMA': 0.2, 'GLM': 1, 'UnobservedComponents': 1, 'UnivariateMotif': 1, 'MultivariateMotif': 0.8,
        'Theta': 1, 'ARDL': 1, 'ARCH': 1, 'ConstantNaive': 1, 'LastValueNaive': 1.5, 'AverageValueNaive': 1, 'GLS': 1, 'SeasonalNaive': 1,
        'VAR': 0.8, 'VECM': 1, 'WindowRegression': 0.5, 'DatepartRegression': 0.8, 'SectionalMotif': 1, 'NVAR': 1, 'MAR': 1, 'RRVAR': 1,
        'KalmanStateSpace': 1, 'MetricMotif': 1, 'Cassandra': 1, 'SeasonalityMotif': 1
    },
    'fast_parallel_no_arima': {
        'ETS': 1, 'FBProphet': 0.8, 'GLM': 1, 'UnobservedComponents': 1, 'UnivariateMotif': 1, 'MultivariateMotif': 0.8, 'Theta': 1,
        'ARDL': 1, 'ARCH': 1, 'ConstantNaive': 1, 'LastValueNaive': 1.5, 'AverageValueNaive': 1, 'GLS': 1, 'SeasonalNaive': 1, 'VAR': 0.8,
        'VECM': 1, 'WindowRegression': 0.5, 'DatepartRegression': 0.8, 'SectionalMotif': 1, 'MAR': 1, 'RRVAR': 1, 'KalmanStateSpace': 1,
        'MetricMotif': 1, 'Cassandra': 1, 'SeasonalityMotif': 1
    },
    'probabilistic': [
        'ARIMA', 'GluonTS', 'FBProphet', 'AverageValueNaive', 'DynamicFactor', 'VAR', 'UnivariateMotif', 'MultivariateMotif',
        'SectionalMotif', 'NVAR', 'Theta', 'ARDL', 'UnobservedComponents', 'DynamicFactorMQ', 'PytorchForecasting', 'ARCH',
        'KalmanStateSpace', 'MetricMotif', 'Cassandra', 'SeasonalityMotif'
    ],
    'multivariate': [
        'VECM', 'DynamicFactor', 'GluonTS', 'RollingRegression', 'WindowRegression', 'VAR', 'MultivariateMotif', 'NVAR',
        'MultivariateRegression', 'SectionalMotif', 'DynamicFactorMQ', 'PytorchForecasting', 'RRVAR', 'MAR', 'TMF', 'LATC', 'Cassandra'
    ],
    'univariate': [
        'MLEnsemble', 'UnivariateRegression', 'SeasonalityMotif', 'FBProphet', 'AverageValueNaive', 'LastValueNaive',
        'DatepartRegression', 'ARCH', 'ETS', 'GLS', 'ConstantNaive', 'SeasonalNaive', 'ARDL', 'MetricMotif', 'Theta', 'GLM',
        'UnobservedComponents', 'KalmanStateSpace', 'NeuralProphet', 'ARIMA', 'UnivariateMotif'
    ],
    'no_params': [
        'LastValueNaive', 'GLS'
    ],
   'recombination_approved': [
        'SeasonalNaive', 'MotifSimulation', 'ETS', 'DynamicFactor', 'VECM', 'VARMAX', 'GLM', 'ARIMA', 'FBProphet', 'GluonTS',
        'RollingRegression', 'VAR', 'TensorflowSTS', 'TFPRegression', 'UnivariateRegression', 'Greykite', 'UnivariateMotif',
        'MultivariateMotif', 'NVAR', 'MultivariateRegression', 'SectionalMotif', 'Theta', 'ARDL', 'NeuralProphet', 'DynamicFactorMQ',
        'PytorchForecasting', 'ARCH', 'RRVAR', 'MAR', 'TMF', 'LATC', 'MetricMotif', 'Cassandra', 'SeasonalityMotif'
    ],
    'no_shared': [
        'ConstantNaive', 'LastValueNaive', 'AverageValueNaive', 'GLM', 'ETS', 'ARIMA', 'FBProphet', 'SeasonalNaive',
        'UnobservedComponents', 'TensorflowSTS', 'GLS', 'UnivariateRegression', 'Greykite', 'UnivariateMotif', 'Theta', 'ARDL',
        'NeuralProphet', 'ARCH', 'KalmanStateSpace', 'MetricMotif', 'SeasonalityMotif'
    ],
    'no_shared_fast': [
        'SeasonalityMotif', 'FBProphet', 'AverageValueNaive', 'LastValueNaive', 'UnivariateMotif', 'MetricMotif', 'Theta', 'ARCH',
        'ETS', 'GLM', 'ConstantNaive', 'UnobservedComponents', 'GLS', 'SeasonalNaive', 'KalmanStateSpace', 'ARDL'
    ],
    'experimental': [
        'MotifSimulation', 'TensorflowSTS', 'ComponentAnalysis', 'TFPRegression'
    ],
    'slow': [
        'MLEnsemble', 'UnivariateRegression', 'RollingRegression', 'FBProphet', 'ARDL', 'MultivariateRegression', 'DynamicFactor',
        'Theta', 'ARCH', 'LATC', 'UnobservedComponents', 'TMF', 'PytorchForecasting', 'DynamicFactorMQ', 'NeuralProphet', 'ARIMA',
        'GluonTS'
    ],
    'gpu': [
        'GluonTS', 'WindowRegression', 'PytorchForecasting'
    ],
    'regressor': [
        'GLM', 'ARIMA', 'FBProphet', 'RollingRegression', 'UnobservedComponents', 'VECM', 'DynamicFactor', 'WindowRegression',
        'VAR', 'DatepartRegression', 'GluonTS', 'UnivariateRegression', 'MultivariateRegression', 'SectionalMotif', 'ARDL',
        'NeuralProphet', 'ARCH', 'Cassandra'
    ],
    'best': [
        'SeasonalityMotif', 'AverageValueNaive', 'WindowRegression', 'DatepartRegression', 'ETS', 'ConstantNaive', 'SeasonalNaive',
        'MAR', 'ARDL', 'VAR', 'Theta', 'GLM', 'UnobservedComponents', 'KalmanStateSpace', 'UnivariateMotif', 'FBProphet',
        'LastValueNaive', 'ARCH', 'GLS', 'RRVAR', 'MultivariateRegression', 'MetricMotif', 'Cassandra', 'VECM', 'SectionalMotif',
        'PytorchForecasting', 'GluonTS', 'MultivariateMotif'
    ],
    'motifs': [
        'UnivariateMotif', 'MultivariateMotif', 'SectionalMotif', 'MotifSimulation', 'MetricMotif', 'SeasonalityMotif'
    ],
    'all_result_path': [
        'UnivariateMotif', 'MultivariateMotif', 'SectionalMotif', 'MetricMotif', 'SeasonalityMotif', 'Motif', 'ARCH',
        'PytorchForecasting'
    ],
    'regressions': [
        'RollingRegression', 'WindowRegression', 'DatepartRegression', 'UnivariateRegression', 'MultivariateRegression'
    ],
    'all_pragmatic': [
        'UnivariateRegression', 'SeasonalityMotif', 'AverageValueNaive', 'WindowRegression', 'DatepartRegression', 'ETS',
        'ConstantNaive', 'SeasonalNaive', 'DynamicFactorMQ', 'MAR', 'ARDL', 'VAR', 'Theta', 'GLM', 'LATC', 'UnobservedComponents',
        'TMF', 'KalmanStateSpace', 'RollingRegression', 'UnivariateMotif', 'FBProphet', 'LastValueNaive', 'DynamicFactor', 'ARCH',
        'GLS', 'RRVAR', 'MotifSimulation', 'NVAR', 'MultivariateRegression', 'MetricMotif', 'Cassandra', 'VECM', 'SectionalMotif',
        'PytorchForecasting', 'NeuralProphet', 'ARIMA', 'GluonTS', 'MultivariateMotif'
    ],
    'update_fit': [
        'MultivariateRegression', 'DatepartRegression', 'GluonTS', 'WindowRegression', 'Cassandra'
    ]
}


# Ticker symbol
ticker_symbol = "AAPL"

# Start-end dates
start_date = "2021-09-24"
end_date = "2023-09-24"

# Fetch data
df = yf.download(ticker_symbol, start=start_date, end=end_date)
df.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
# Keep 1D time series "close"
df = df[['close']]
df = df.rename_axis("date")
df.index.name = "date"
df.reset_index(inplace=True)
df['date'] = pd.to_datetime(df['date'])

# Split training and test
train_size = int(len(df) * 0.8)  # 80% training

train_data = df[:train_size]
test_data = df[train_size:]

# Choose a random sample of models
model_list = []
all_values = [value for values in model_dict.values() for value in values]
number_models = 1  # Desired number of models

if number_models <= len(all_values):
    model_list = random.sample(all_values, number_models)
else:
    model_list = all_values



model = AutoTS(
    forecast_length=10,
    frequency='infer',
    prediction_interval=0.95,
    ensemble=['simple', 'horizontal-min'],
    max_generations=5,
    num_validations=2,
    validation_method='seasonal 168',
    model_list=model_list,
    transformer_list='all',
    models_to_validate=0.2,
    drop_most_recent=1,
    n_jobs='auto',
)
model = model.fit(train_data, date_col='date', value_col='close')
test_prediction = model.predict() 

# Plot prediction
test_prediction.plot(model.df_wide_numeric,
                series=model.df_wide_numeric.columns[0],
                start_date="2022-09-01"
)
plt.show()

# Save model
model.export_template(
    "autots_model.csv",
    models="best",
    max_per_model_class=1,
    include_results=True
)
