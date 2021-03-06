import pickle
from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


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

        lr = RandomForestRegressor(n_estimators=50, max_depth=4, min_samples_split=4,
                                   max_features=0.5)

        lr.fit(data, y)

        # Serialize model using pickle
        date = datetime.now().strftime("%Y%m%d_%H%M")
        filename = 'src/models/reg_{}.pickle'.format(date)
        with open(filename, 'wb') as f:
            pickle.dump(lr, f, pickle.HIGHEST_PROTOCOL)
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

