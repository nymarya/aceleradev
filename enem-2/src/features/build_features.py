import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, OneHotEncoder, \
    FunctionTransformer
from sklearn.pipeline import make_pipeline
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

        print('Droping columns with missing values')
        df = df.loc[:, (df.isna().sum() / df.shape[0]) * 100 < 50]
        print('Dropping columns with constant values')
        df = df.loc[:, df.nunique() > 1]
        print('Dropping columns correlated')
        cols = df.columns[np.isin(df.columns, self.correlated)]
        df.drop(columns=cols)

        # Preprocess the text data: get_text_data
        get_text_data = FunctionTransformer(
            lambda x: x.select_dtypes(include='object'),
            validate=False)

        # Preprocess the numeric data
        get_numeric_data = FunctionTransformer(
            lambda x: x.select_dtypes(include='number'),
            validate=False)

        # Build pipeline
        pl = Pipeline([
            ('union', FeatureUnion(
                transformer_list=[
                    ('numeric_features', Pipeline([
                        ('selector', get_numeric_data),
                        ('imp', SimpleImputer(strategy='constant', fill_value=-1)),
                        ('scaler', StandardScaler())
                    ])),
                    ('text_features', Pipeline([
                        ('selector', get_text_data),
                        ('imp',
                         SimpleImputer(strategy='constant', fill_value='N/A',
                                       missing_values=np.nan)),

                        ('enc', OneHotEncoder(drop='first'))
                    ]))
                ]
            )),
            ('red', TruncatedSVD())
        ])

        print('Transforming')
        if training:
            new_data = pl.fit_transform(df)
            return new_data, y
        else:
            return pl.fit_transform(df)
