import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, RFE, chi2
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, \
    FunctionTransformer, OrdinalEncoder, Normalizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
import numpy as np


class Preprocessing:

    def __init__(self):
        # Correlated colums
        self.correlated = ['CO_MUNICIPIO_ESC', 'CO_MUNICIPIO_RESIDENCIA',
                           'CO_MUNICIPIO_NASCIMENTO', 'CO_MUNICIPIO_PROVA']

        self.feature_names = None
        self.std_scaler = None
        self.categoric_features = None
        self.numeric_features = None
        self.enc = None
        self.scaler = None
        self.train_features = None

    def process(self, df: pd.DataFrame, target: str = None, training: bool = True):
        """ Process data.
            Attributes
            ---------
            df: pd.DataFrame
                Dataframe containing data
            target: str
                Name of the column
            training: bool
                Flag indicating whether the processing is used for training
                or not

            Return
            ------
            Transformed data
        """
        if training:
            y = df[target].fillna(-1)
            df.drop(columns=[target], inplace=True)

        df = self.basic_cleaning(df)

        # Preprocess the text data: get_text_data
        get_text_data = FunctionTransformer(
            lambda x: x.select_dtypes(include='object'),
            validate=False)

        # Preprocess the numeric data
        get_numeric_data = FunctionTransformer(
            lambda x: x.select_dtypes(include='number'),
            validate=False)

        estimator=LinearRegression()

        # Build pipeline
        pl = Pipeline([
            ('union', FeatureUnion(
                transformer_list=[
                    ('numeric_features', Pipeline([
                        ('selector', get_numeric_data),
                        ('imp', SimpleImputer(strategy='constant',
                                              fill_value=-1)),
                        ('scaler', Normalizer())
                    ])),
                    ('text_features', Pipeline([
                        ('selector', get_text_data),
                        ('imp', SimpleImputer(strategy='constant',
                                              fill_value='N/A')),
                        ('enc', OneHotEncoder())
                    ]))
                ]
            ))

        ])

        print('Transforming')
        if training:
            new_data = pl.fit_transform(df)
            return new_data, y
        else:
            return pl.fit_transform(df)

    def basic_cleaning(self, df: pd.DataFrame):
        """ Basic pre-processing steps.
            Attributes
            ---------
            df: pd.DataFrame
                Dataframe containing data

            Return
            ------
            Transformed data
        """
        print('Droping columns with missing values')
        df = df.loc[:, (df.isna().sum() / df.shape[0]) * 100 < 50]
        print('Dropping columns with constant values')
        df = df.loc[:, df.nunique() > 1]
        print('Dropping columns correlated')
        cols = df.columns[np.isin(df.columns, self.correlated)]
        df.drop(columns=cols)

        return df

    def remove_outlier(self, df: pd.DataFrame):
        pass
