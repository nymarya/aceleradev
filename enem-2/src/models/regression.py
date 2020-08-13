from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import RFE, SelectKBest, chi2, f_classif
from sklearn.linear_model import LinearRegression, Ridge, TheilSenRegressor
import pickle
from datetime import datetime
import pandas as pd
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

        lr = LinearRegression()

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

