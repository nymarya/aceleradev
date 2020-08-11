import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, RFE, chi2, f_classif
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, \
    FunctionTransformer, OrdinalEncoder, Normalizer, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
import numpy as np
import pickle

from sklearn.svm import SVR, SVC


class Preprocessing:

    def __init__(self):
        # Correlated colums
        self.correlated = ['CO_MUNICIPIO_ESC', 'CO_MUNICIPIO_RESIDENCIA',
                           'CO_MUNICIPIO_NASCIMENTO', 'CO_MUNICIPIO_PROVA',
                           'NO_MUNICIPIO_RESIDENCIA', 'NO_MUNICIPIO_NASCIMENTO']

        self.pl = None
        self.cols = None
        self.features_selected = None

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
            df = self.basic_cleaning(df)
            cols = self.filter_correlation(df, target)
            categorical = df.select_dtypes(include='object').columns.tolist()
            print(categorical)
            df = df.loc[:, cols + categorical]
            y = df[target].fillna(-1)
            df.drop(columns=[target], inplace=True)

            self.cols = df.columns

        # Preprocess the text data: get_text_data
        get_text_data = FunctionTransformer(
            lambda x: x.select_dtypes(include='object').fillna('N/A').values,
            validate=False)

        # Preprocess the numeric data
        get_numeric_data = FunctionTransformer(
            lambda x: x.select_dtypes(include='number').fillna(-1).values,
            validate=False)

        estimator = SVC(kernel='linear', C=1)

        print('Transforming')
        # Build pipeline
        pl = Pipeline([
            ('union', FeatureUnion(
                transformer_list=[
                    ('numeric_features', Pipeline([
                        ('selector', get_numeric_data),
                        # ('imp', SimpleImputer(strategy='constant',
                        #                       fill_value=-1)),
                        ('scaler', Normalizer())
                    ])),
                    ('text_features', Pipeline([
                        ('selector', get_text_data),
                        # ('imp', SimpleImputer(strategy='constant',
                        #                       fill_value='N/A')),
                        ('enc', OrdinalEncoder())
                    ]))
                ]
            )),
            ('red', SelectKBest(f_classif, k=10))
        ])

        print(df.columns)
        if training:
            print(df.shape)
            self.pl = pl
            new_data = self.pl.fit_transform(df, y)

            return new_data, y
        else:
            print(df.shape)
            return self.pl.transform(df[self.cols])

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
        df.drop(columns=cols, inplace=True)
        df.drop(columns=['NU_INSCRICAO'], inplace=True)

        return df

    def filter_correlation(self, df: pd.DataFrame, target: str):
        corr = df.corr().loc[:, target]

        return corr[corr < -0.2].index.tolist() + corr[corr > 0.5].index.tolist()
