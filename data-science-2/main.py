#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns


# In[3]:


# %matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[4]:


athletes = pd.read_csv("athletes.csv")


# In[5]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[6]:


# Sua análise começa aqui.


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[7]:


def q1():
    # Retorne aqui o resultado da questão 1.
    sample = get_sample(athletes, 'height', 3000)
    stat, p = sct.shapiro(sample)

    # A hipótese não é rejeitada (ie, a amostra pode ter sido
    # obtida de uma distribuição normal) se p > 0.05
    return p > 0.05

q1()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# In[8]:


athletes.height.plot(kind='hist', bins=25)


# O gráfico mostra uma distribuição semelhante a uma distribuição normal

# In[9]:


sct.probplot(athletes.height, dist="norm", plot=plt)
# plt.show()


# O gráfico mostra que a amostra possui pequenos graus de liberdade

# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[19]:


def q2():
    # Retorne aqui o resultado da questão 2.
    sample = get_sample(athletes, 'height', 3000)
    stat, p = sct.jarque_bera(sample)

    # A hipótese não é rejeitada (ie, a amostra pode ter sido
    # obtida de uma distribuição normal) se p > 0.05
    return float(p) > 0.05

q2()


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[11]:


def q3():
    # Retorne aqui o resultado da questão 3.
    sample = get_sample(athletes, 'weight', 3000)
    stat, p = sct.normaltest(sample)

    # A hipótese não é rejeitada (ie, a amostra pode ter sido
    # obtida de uma distribuição normal) se p > 0.05
    return float(p) > 0.05

q3()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# In[12]:


athletes.weight.plot(kind='hist', bins=25)


# In[13]:


athletes.weight.plot(kind='box')


# Pelos gráfico é claro que a distribuição não é gaussiana. No histograma, há um
# notável deslocamento para a esquerda, enquanto no boxplot existem muitos "outliers".

# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[14]:


def q4():
    # Retorne aqui o resultado da questão 4.
    sample = get_sample(athletes, 'weight', 3000)
    log_sample = np.log(sample)
    stat, p = sct.normaltest(log_sample)

    # A hipótese não é rejeitada (ie, a amostra pode ter sido
    # obtida de uma distribuição normal) se p > 0.05
    return float(p) > 0.05

q4()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[15]:


# Recupera amostras

bra_heights = athletes.query('nationality == "BRA"').height
usa_heights = athletes.query('nationality == "USA"').height
can_heights = athletes.query('nationality == "CAN"').height

def q5():
    # Retorne aqui o resultado da questão 5.

    equal_var = len(bra_heights) == len(usa_heights)
    # Realiza teste
    stat, p = sct.ttest_ind(bra_heights, usa_heights, equal_var=equal_var,
                            nan_policy='omit')

    # A hipótese não é rejeitada (ie, a amostra pode ter sido
    # obtida de uma distribuição normal) se p > 0.05
    return float(p) > 0.05


q5()


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[16]:


def q6():
    # Retorne aqui o resultado da questão 6.

    equal_var = len(bra_heights) == len(can_heights)
    # Realiza teste
    stat, p = sct.ttest_ind(bra_heights, can_heights, equal_var=equal_var,
                            nan_policy='omit')

    # A hipótese é rejeitada (ie, as médias não não iguais) se
    # p < 0.05
    return float(p) > 0.05


q6()


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[17]:


def q7():
    # Retorne aqui o resultado da questão 7.

    equal_var = len(usa_heights) == len(can_heights)
    # Realiza teste
    stats, p= sct.ttest_ind(usa_heights, can_heights, equal_var=equal_var,
                            nan_policy='omit')

    return float(round(p, 8))


q7()


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?
