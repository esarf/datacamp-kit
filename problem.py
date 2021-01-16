import os
import pandas as pd
import rampwf as rw
from rampwf.workflows import FeatureExtractorRegressor
from sklearn.model_selection import ShuffleSplit
import numpy as np
from sklearn.metrics import mean_squared_error



from rampwf.score_types.base import BaseScoreType


_train = 'train.csv'
_test = 'test.csv'

quick_mode = os.getenv('RAMP_TEST_MODE', 0)

if(quick_mode):
    _train = 'train_small.csv'
    _test = 'test_small.csv'

problem_title = 'Rents prices in Paris'
_target_column_names = ['ref']

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression(
    label_names=['ref'])
# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorRegressor()

class RMSE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='RMSE', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

score_types = [
    RMSE()
]


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_names].values
    X_df = data.drop('ref', axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = _train
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = _test
    return _read_data(path, f_name)


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    return cv.split(X, y)
