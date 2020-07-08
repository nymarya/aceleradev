#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[4]:


black_friday.info()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[5]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape

q1()


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[6]:


query = 'Gender == "F" and Age=="26-35"'
def q2():
    # Retorne aqui o resultado da questão 2.
    return int(black_friday.query(query)['User_ID'].count())


black_friday.query(query)['User_ID'].count()


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.
# 
# 

# In[7]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday.User_ID.nunique()

black_friday.User_ID.nunique()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[8]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return black_friday.dtypes.nunique()

black_friday.dtypes.nunique()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[9]:


def q5():
    # Retorne aqui o resultado da questão 5.
    total_count = len(black_friday)
    na_count = total_count - len(black_friday.dropna())
    return na_count / total_count

q5()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[10]:


def q6():
    # Retorne aqui o resultado da questão 6.
    count = black_friday.isna().sum().max()
    return int(count)

q6()


# 
# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[11]:


def q7():
    # Retorne aqui o resultado da questão 7.
    most_frequent = black_friday.Product_Category_3.value_counts()

    return most_frequent.idxmax()

q7()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[12]:


def q8():
    # Retorne aqui o resultado da questão 8.
    purchase = black_friday.Purchase
    purchase_max = purchase.max()
    purchase_min = purchase.min()
    norm_purchase = (purchase - purchase_min) / (purchase_max - purchase_min)

    return float(norm_purchase.mean())

q8()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[13]:


def q9():
    # Retorne aqui o resultado da questão 9.
    purchase = black_friday.Purchase
    standard_purchase = ( purchase - purchase.mean()) / purchase.std()

    return int(standard_purchase.between(-1, 1).sum())

q9()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[14]:


def q10():
    # Retorne aqui o resultado da questão 10.
    # Get rows with null values in Product_Category_2
    p2_isna = black_friday.Product_Category_2.isna()
    p2_nulls = black_friday.loc[p2_isna, :]

    p3_isna = p2_nulls.Product_Category_3.isna()

    return bool(p3_isna.sum() == len(p2_nulls))

q10()

