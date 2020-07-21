#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk

from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


# In[2]:


# Algumas configurações para o matplotlib.
# %matplotlib inline
#
# from IPython.core.pylabtools import figsize
#
#
# figsize(12, 8)
#
# sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


countries.info()


# In[6]:


# Fixing float type columns
int_cols = countries.select_dtypes(include=['object']).columns
numeric = int_cols[2:]
def fix_types(col):
    col = col.str.replace(',', '.')
    col = col.astype(float)

    return col


countries[numeric] = countries[numeric].apply(fix_types)

countries[['Country', 'Region']] = countries[['Country', 'Region']]    .apply(lambda x: x.str.strip())

countries.head()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[7]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return countries.Region.sort_values().unique().tolist()

q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[8]:


def q2():
    X = countries.Pop_density.values.reshape(-1, 1)
    est = KBinsDiscretizer(n_bins=10, encode='ordinal',
                                            strategy='quantile')
    bins_popdensity = est.fit(X)

    # 90% is in the second to last position
    upper_limit = bins_popdensity.bin_edges_[0][-2]
    return int((countries.Pop_density > upper_limit).sum())


q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[9]:


def q3():
    nfeatures_region = countries.Region.nunique()
    nfeatures_climate = countries.Climate.nunique()
    return  nfeatures_region + nfeatures_climate + 1

q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[10]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[11]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


# In[34]:


def q4():
    # Filter cols and get indexes
    numeric_cols = countries.select_dtypes(include=['float64', 'int64'])
    cols = [countries.columns.get_loc(col) for col in numeric_cols]

    # Build pipeline
    pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())

    # Fit and tranforsm
    pipe.fit(countries.iloc[:, cols])
    X = [ test_country[i] for i in cols]
    result = pipe.transform([X])

    # Get old column index and subtract 2 (initial columns
    # that were not used)
    arable_col = countries.columns.get_loc('Arable') - 2
    return float(result[0][arable_col].round(3))


q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[13]:


def q5():
    # Retorne aqui o resultado da questão 4.
    quant1, quant3 = countries.Net_migration.quantile([.25, .75])
    iqr = quant3 - quant1

    outliers_under = countries.Net_migration < (quant1 - 1.5 * iqr)
    outliers_above = countries.Net_migration > (quant3 + 1.5 * iqr)

    return (int(outliers_under.sum()), int(outliers_above.sum()), False)


q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[14]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories,
                               shuffle=True, random_state=42)


# In[15]:


def q6():
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(newsgroup['data'])
    index = vectorizer.vocabulary_['phone']
    countings = X.toarray().sum(axis=0)
    return int(countings[index])


q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[16]:


def q7():
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(newsgroup['data'])
    index = vectorizer.vocabulary_['phone']
    countings = X.toarray().sum(axis=0)
    return float(round(countings[index], 3))


q7()

