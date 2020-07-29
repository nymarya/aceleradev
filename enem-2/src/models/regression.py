from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.linear_model import LinearRegression, Ridge
import pickle
from datetime import datetime
import pandas as pd
from numpy import arange
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR


class Regression:

    def __init__(self):
        self.model = None

    def train(self, data: pd.DataFrame, y):
        """ Train model.

            Attributes
            ---------
            data: pd.DataFrame
                Dataframe containing data
            y: pd.Series
                Target column
        """

        pl = Pipeline([
            # the reduce_dim stage is populated by the param_grid
            ('red', 'passthrough'),
            ('classify', LinearRegression())
        ])

        n_features_options = [10, 20, 30, 40]
        estimator = SVR(kernel="linear")

        param_grid = [
            {
                'red': [TruncatedSVD()],
                'red__n_components': n_features_options
            },
            {
                'red': [RFE(estimator, step=10, verbose=1)],
                'red__n_features_to_select': n_features_options
            },
            {
                'red': [SelectKBest(chi2)],
                'red__k': n_features_options
            },
        ]

        grid = GridSearchCV(pl, n_jobs=1, param_grid=param_grid, verbose=10)
        grid.fit(data, y)

        # Print best estimator
        print("Best estimator is:")
        print(grid.best_estimator_)

        # Serialize model using pickle
        date = datetime.now().strftime("%Y%m%d_%H%M")
        filename = 'src/models/reg_{}.pickle'.format(date)
        with open(filename, 'wb') as f:
            pickle.dump(grid, f, pickle.HIGHEST_PROTOCOL)
            print("Model save at: " + filename)
            self.model = filename

    def predict(self, data: pd.DataFrame):
        """ Make a prediction using the model.

            Attributes
            ---------
            data: pd.DataFrame
                Dataframe containing data
        """
        with open(self.model, 'rb') as f:
            reg = pickle.load(f)
            response = reg.predict(data)

            return response

    def score(self, data: pd.DataFrame, y: pd.Series):
        """ Get score from model.

            Attributes
            ---------
            data: pd.DataFrame
                Dataframe containing data
            y: pd.Series
                Target column
        """
        with open(self.model, 'rb') as f:
            reg = pickle.load(f)
            return reg.score(data, y)

