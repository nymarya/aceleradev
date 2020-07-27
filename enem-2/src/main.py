from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV

from features.build_features import Preprocessing
from models.regression import Regression
import pandas as pd
import numpy as np

enem_df = pd.read_csv('train.csv', index_col='Unnamed: 0')

features = Preprocessing()
train_data, train_target = features.process(enem_df, target='NU_NOTA_MT')
print(train_data.shape)
X_train, X_test, y_train, y_test = train_test_split(train_data, train_target,
                                                    test_size=0.4,
                                                    random_state=42)


# Training
reg = Regression()
reg.train(X_train, y_train)
print(reg.score(X_test, y_test))

# reg2 = Ridge()
# parameters = {'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
#               'tol': np.arange(0.00001, 0.1, 0.02)}
# clf = GridSearchCV(reg2, parameters)
# clf.fit(X_train, y_train)
# print(clf.best_score_)

# Predict
test_df = pd.read_csv('test.csv')
# print(test_df.columns)
X_test = features.process(test_df, training=False)
#
grades = reg.predict(X_test)


answers = pd.DataFrame()
answers['NU_INSCRICAO'] = test_df.NU_INSCRICAO
answers['NU_NOTA_MT'] = grades

answers.to_csv('answer.csv', index=False)
